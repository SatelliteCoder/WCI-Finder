from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_jsonable(value.item())
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def first_number(arr: np.ndarray, default: float = float("nan")) -> float:
    flat = np.asarray(arr).ravel(order="F")
    return float(flat[0]) if flat.size else default


def moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = values.astype(float, copy=False)
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    kernel = np.ones(window, dtype=float)
    summed = np.convolve(clean, kernel, mode="same")
    counts = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.full(values.shape, np.nan, dtype=float)
    ok = counts > 0
    out[ok] = summed[ok] / counts[ok]
    return out


def wrap180(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def product_id_from_all_name(survey_id: str, all_name: str) -> str:
    base = Path(all_name).stem
    match = re.match(r"^(\d{4})_(\d{8})_\d+_(.+)$", base)
    if match:
        line, date, platform = match.groups()
        return f"{date}_{line}_{platform}"
    clean_survey = re.sub(r"[^\w]+", "_", survey_id)
    clean_base = re.sub(r"[^\w]+", "_", base)
    return f"{clean_survey}_{clean_base}"


def utm_epsg_from_tmproj_or_lon(tmproj: str, lon: float | None = None) -> int:
    match = re.search(r"utm(\d+)", str(tmproj).lower())
    if match:
        return 32600 + int(match.group(1))
    if lon is not None and math.isfinite(lon):
        return 32600 + int((lon + 180.0) // 6.0) + 1
    return 32633


def locate_all_file(data_root: Path, survey_id: str, all_name: str) -> Path:
    direct = data_root / survey_id / all_name
    if direct.exists():
        return direct
    matches = list((data_root / survey_id).rglob(Path(all_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"ALL file not found under {data_root / survey_id}: {all_name}")


class RawKongsbergCache:
    def __init__(self, all_file: Path, wcd_file: Path) -> None:
        vendor_dir = ROOT / "src" / "vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        from kongsberg_par3 import AllRead

        if not wcd_file.exists():
            raise FileNotFoundError(f"WCD file not found next to ALL file: {wcd_file}")

        self.all_file = all_file
        self.wcd_file = wcd_file
        self.cache_dir = all_file.parent
        self.tmproj = "utm33N"
        self.ellips = "wgs84"
        self.wcd_reader = AllRead(str(wcd_file))
        self.wcd_reader.mapfile(show_progress=False)
        if "107" not in self.wcd_reader.map.packdir:
            raise RuntimeError(f"No Kongsberg water-column datagram 107 found in {wcd_file}")

        self.n_pings = int(getattr(self.wcd_reader.map, "numwc", 0) or len(sorted(set(self.wcd_reader.map.packdir["107"][:, 3]))))
        if self.n_pings <= 0:
            raise RuntimeError(f"No complete water-column pings found in {wcd_file}")

        self.wc_time = self._water_column_times()
        self.ping_counter = np.zeros(self.n_pings, dtype=float)
        self.sound_speed = np.full(self.n_pings, 1500.0, dtype=float)
        self.sample_freq = np.full(self.n_pings, np.nan, dtype=float)
        self.n_samples = 0
        self.n_beams = 0
        self.beam_angles_rad: np.ndarray
        self.start_sample_number: np.ndarray
        self.bottom_sample_wc: np.ndarray
        self._scan_water_column_geometry()
        self.easting, self.northing, self.height, self.heading, self.speed = self._load_navigation()

    def close(self) -> None:
        try:
            self.wcd_reader.close()
        except Exception:
            pass

    def _water_column_times(self) -> np.ndarray:
        rows = np.asarray(self.wcd_reader.map.packdir["107"], dtype=float)
        times_by_counter: dict[int, float] = {}
        for row in rows:
            counter = int(row[3])
            times_by_counter.setdefault(counter, float(row[1]))
        return np.asarray([times_by_counter[k] for k in sorted(times_by_counter)], dtype=float)

    def _scan_water_column_geometry(self) -> None:
        ping_meta: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, tuple[int, int]]] = []
        max_samples = 0
        max_beams = 0
        print(f"[0/6] Scanning raw Kongsberg WCD geometry: {self.n_pings} pings")
        for idx in range(self.n_pings):
            packet = self.wcd_reader.getwatercolumn(idx)
            if packet is None:
                continue
            amp_shape = tuple(int(v) for v in packet.ampdata.shape)
            rx = packet.rx
            angles = np.deg2rad(np.asarray(rx["BeamPointingAngle"], dtype=float))
            starts = np.asarray(rx["StartRangeSample#"], dtype=float)
            detected = np.asarray(rx["DetectedRange"], dtype=float)
            max_samples = max(max_samples, amp_shape[0])
            max_beams = max(max_beams, amp_shape[1])
            ping_meta.append(
                (
                    angles,
                    starts,
                    detected,
                    float(packet.header["SoundSpeed"]),
                    float(packet.header["SamplingFrequency"]),
                    int(packet.header["PingCounter"]),
                    amp_shape,
                )
            )
        if not ping_meta:
            raise RuntimeError(f"Unable to decode any complete WCD ping from {self.wcd_file}")

        self.n_pings = len(ping_meta)
        self.wc_time = self.wc_time[: self.n_pings]
        self.n_samples = int(max_samples)
        self.n_beams = int(max_beams)
        self.beam_angles_rad = np.full((self.n_pings, self.n_beams), np.nan, dtype=float)
        self.start_sample_number = np.zeros((self.n_pings, self.n_beams), dtype=float)
        self.bottom_sample_wc = np.full((self.n_pings, self.n_beams), np.nan, dtype=float)
        self.ping_counter = np.zeros(self.n_pings, dtype=float)
        self.sound_speed = np.full(self.n_pings, 1500.0, dtype=float)
        self.sample_freq = np.full(self.n_pings, np.nan, dtype=float)

        for idx, (angles, starts, detected, sound_speed, sample_freq, ping_counter, _shape) in enumerate(ping_meta):
            n_beams = min(self.n_beams, angles.size)
            self.beam_angles_rad[idx, :n_beams] = angles[:n_beams]
            self.start_sample_number[idx, :n_beams] = starts[:n_beams]
            self.bottom_sample_wc[idx, :n_beams] = detected[:n_beams]
            self.sound_speed[idx] = sound_speed
            self.sample_freq[idx] = sample_freq
            self.ping_counter[idx] = ping_counter

        missing_angle = ~np.isfinite(self.beam_angles_rad)
        if missing_angle.any():
            fallback = np.nanmedian(self.beam_angles_rad, axis=0)
            fallback[~np.isfinite(fallback)] = 0.0
            self.beam_angles_rad[missing_angle] = np.take(fallback, np.where(missing_angle)[1])

    def _load_navigation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vendor_dir = ROOT / "src" / "vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        from kongsberg_par3 import AllRead

        nav_file = self.all_file if self.all_file.exists() else self.wcd_file
        with AllRead(str(nav_file)) as reader:
            reader.mapfile(show_progress=False)
            pos_time: list[float] = []
            lat: list[float] = []
            lon: list[float] = []
            speed: list[float] = []
            heading_pos: list[float] = []
            if "80" in reader.map.packdir:
                for rec_idx in range(reader.map.packdir["80"].shape[0]):
                    rec = reader.getrecord(80, rec_idx)
                    pos_time.append(float(rec.time))
                    lat.append(float(rec.header["Latitude"]))
                    lon.append(float(rec.header["Longitude"]))
                    speed.append(float(rec.header["Speed"]))
                    heading_pos.append(float(rec.header["Heading"]))
            if not pos_time:
                raise RuntimeError(f"No Kongsberg position datagram 80 found in {nav_file}")

            att_time: list[float] = []
            att_heading: list[float] = []
            if "65" in reader.map.packdir:
                for rec_idx in range(reader.map.packdir["65"].shape[0]):
                    rec = reader.getrecord(65, rec_idx)
                    data = rec.data
                    att_time.extend([float(v) for v in data["Time"]])
                    att_heading.extend([float(v) for v in data["Heading"]])

        pos_time_arr = np.asarray(pos_time, dtype=float)
        order = np.argsort(pos_time_arr)
        pos_time_arr = pos_time_arr[order]
        lat_arr = np.asarray(lat, dtype=float)[order]
        lon_arr = np.asarray(lon, dtype=float)[order]
        speed_arr = np.asarray(speed, dtype=float)[order]
        heading_pos_arr = np.asarray(heading_pos, dtype=float)[order]

        wc_time = self.wc_time
        interp_lat = np.interp(wc_time, pos_time_arr, lat_arr)
        interp_lon = np.interp(wc_time, pos_time_arr, lon_arr)
        interp_speed = np.interp(wc_time, pos_time_arr, speed_arr)

        if att_time:
            att_time_arr = np.asarray(att_time, dtype=float)
            order = np.argsort(att_time_arr)
            interp_heading = np.interp(wc_time, att_time_arr[order], np.asarray(att_heading, dtype=float)[order])
        else:
            interp_heading = np.interp(wc_time, pos_time_arr, heading_pos_arr)

        epsg = utm_epsg_from_tmproj_or_lon(self.tmproj, float(np.nanmedian(interp_lon)))
        self.tmproj = f"utm{epsg - 32600}N"
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        easting, northing = transformer.transform(interp_lon, interp_lat)
        height = np.zeros(self.n_pings, dtype=float)
        return (
            np.asarray(easting, dtype=float),
            np.asarray(northing, dtype=float),
            height,
            np.asarray(interp_heading, dtype=float),
            np.asarray(interp_speed, dtype=float),
        )

    def inter_sample_distance(self, ping: int) -> float:
        idx = max(0, min(self.n_pings - 1, ping - 1))
        ss = float(self.sound_speed[idx]) if idx < self.sound_speed.size else 1500.0
        sf = float(self.sample_freq[idx]) if idx < self.sample_freq.size else float("nan")
        if not math.isfinite(sf) or sf <= 0:
            return 0.65
        return ss / (2.0 * sf)

    def ping_wcd(self, ping: int) -> np.ndarray:
        idx = int(np.clip(ping - 1, 0, self.n_pings - 1))
        packet = self.wcd_reader.getwatercolumn(idx)
        if packet is None:
            raise IndexError(f"Ping {ping} outside raw WCD range")
        raw = np.asarray(packet.ampdata, dtype=np.float32)
        out = np.full((self.n_samples, self.n_beams), np.nan, dtype=np.float32)
        s = min(self.n_samples, raw.shape[0])
        b = min(self.n_beams, raw.shape[1])
        out[:s, :b] = raw[:s, :b]
        return out

    def range_grid_for_ping(self, ping: int) -> np.ndarray:
        samples = np.arange(1, self.n_samples + 1, dtype=float)[:, None]
        starts = self.start_sample_number[ping - 1, :][None, :]
        return (samples + starts) * self.inter_sample_distance(ping)

    def range_axis(self) -> np.ndarray:
        step = float(np.nanmedian([self.inter_sample_distance(i) for i in range(1, self.n_pings + 1)]))
        return np.arange(1, self.n_samples + 1, dtype=float) * step

    def wci_coordinates(self, ping: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ranges = self.range_grid_for_ping(ping)
        angles = self.beam_angles_rad[ping - 1, :][None, :]
        across = ranges * np.sin(angles)
        height = -ranges * np.cos(angles)
        return across, height, ranges

    def projected_coordinates(self, ping: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        across, height, _ = self.wci_coordinates(ping)
        idx = ping - 1
        heading_rad = math.radians(float(self.heading[idx]))
        starboard_e = math.cos(heading_rad)
        starboard_n = -math.sin(heading_rad)
        easting = float(self.easting[idx]) + across * starboard_e
        northing = float(self.northing[idx]) + across * starboard_n
        h = float(self.height[idx]) + height
        return easting, northing, h

    def bottom_sample(self, ping: int, profile: np.ndarray) -> int:
        row = self.bottom_sample_wc[ping - 1, :]
        finite = row[np.isfinite(row) & (row > 0)]
        if finite.size:
            value = int(round(float(np.nanmedian(finite))))
            if 8 <= value <= self.n_samples:
                return value
        return estimate_bottom_sample(profile)


def estimate_bottom_sample(profile: np.ndarray) -> int:
    profile = np.asarray(profile, dtype=float)
    if not np.isfinite(profile).any():
        return profile.size
    smooth = moving_mean(moving_mean(profile, 7), 9)
    start = max(7, int(round(profile.size * 0.12)))
    end = max(start + 1, profile.size - 4)
    segment = smooth[start:end]
    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return profile.size
    baseline = float(np.nanmedian(finite))
    high = float(np.nanpercentile(finite, 95))
    threshold = baseline + 0.55 * max(0.5, high - baseline)
    hits = np.flatnonzero(segment >= threshold)
    if hits.size:
        return int(start + hits[0] + 1)
    return int(start + int(np.nanargmax(segment)) + 1)


def topn_mean(data: np.ndarray, n_top: int = 5) -> np.ndarray:
    clean = np.where(np.isfinite(data), data, -np.inf)
    n_top = min(n_top, clean.shape[1])
    top = np.partition(clean, -n_top, axis=1)[:, -n_top:]
    top[top == -np.inf] = np.nan
    return np.nanmean(top, axis=1)


def linear_power_mean_db(data: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(data)
    power = np.zeros(data.shape, dtype=np.float64)
    power[finite] = 10.0 ** (np.clip(data[finite], -120.0, 60.0) / 10.0)
    counts = np.sum(finite, axis=axis)
    summed = np.sum(power, axis=axis)
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = 10.0 * np.log10(summed[ok] / counts[ok])
    return out


def build_detection(cache: RawKongsbergCache, threshold_k: float, bottom_guard: int, max_regions: int, product_id: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    n_samples = cache.n_samples
    n_pings = cache.n_pings
    echo = np.full((n_samples, n_pings), np.nan, dtype=np.float32)
    peak_beam = np.full((n_samples, n_pings), np.nan, dtype=np.float32)
    bottom = np.full(n_pings, np.nan, dtype=np.float32)
    valid_water = np.zeros((n_samples, n_pings), dtype=bool)

    print(f"[1/6] Reading water-column data: {n_samples} samples x {cache.n_beams} beams x {n_pings} pings")
    for ping in range(1, n_pings + 1):
        wcd = cache.ping_wcd(ping)
        profile = linear_power_mean_db(wcd, axis=1)
        bottom_sample = cache.bottom_sample(ping, profile)
        bottom[ping - 1] = bottom_sample
        water_end = max(1, min(n_samples, bottom_sample - bottom_guard))
        valid_water[:water_end, ping - 1] = True
        masked = wcd.copy()
        masked[water_end:, :] = np.nan
        echo[:, ping - 1] = linear_power_mean_db(masked, axis=1)
        finite_rows = np.isfinite(masked).any(axis=1)
        if finite_rows.any():
            filled = np.where(np.isfinite(masked), masked, -np.inf)
            beams = np.argmax(filled, axis=1) + 1
            peak_beam[finite_rows, ping - 1] = beams[finite_rows]
        if ping % 100 == 0 or ping == n_pings:
            print(f"      processed ping {ping}/{n_pings}")

    bg = np.nanmedian(echo, axis=1)[:, None]
    mad = np.nanmedian(np.abs(echo - bg), axis=1)[:, None]
    sigma = 1.4826 * mad
    sigma[~np.isfinite(sigma) | (sigma < 0.75)] = 0.75
    score = (echo - bg) / sigma
    score[~valid_water] = np.nan
    mask = (score >= threshold_k) & valid_water

    labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    regions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    print(f"[2/6] Detecting anomalous echo components: {n_labels} raw components")
    for label_id in range(1, n_labels + 1):
        sample_idx0, ping_idx0 = np.where(labels == label_id)
        if sample_idx0.size == 0:
            continue
        samples = sample_idx0 + 1
        pings = ping_idx0 + 1
        area = int(samples.size)
        ping_span = int(pings.max() - pings.min() + 1)
        sample_span = int(samples.max() - samples.min() + 1)
        if area < 8 or ping_span < 3 or sample_span < 2 or sample_span > 90:
            continue
        density = area / float(ping_span * sample_span)
        if density < 0.08:
            continue
        clearance = float(np.nanmedian(bottom[ping_idx0] - samples))
        if not math.isfinite(clearance) or clearance < bottom_guard:
            continue
        vals = score[sample_idx0, ping_idx0]
        amps = echo[sample_idx0, ping_idx0]
        best = int(np.nanargmax(vals))
        best_sample = int(samples[best])
        best_ping = int(pings[best])
        beam = int(round(float(peak_beam[best_sample - 1, best_ping - 1])))
        if beam < 1 or beam > cache.n_beams:
            continue
        records.append(
            {
                "samples": samples,
                "pings": pings,
                "area_pixels": area,
                "ping_span": ping_span,
                "sample_span": sample_span,
                "density": density,
                "bottom_clearance_samples": clearance,
                "peak_score_mad": float(np.nanmax(vals)),
                "mean_score_mad": float(np.nanmean(vals)),
                "peak_amplitude_db": float(amps[best]),
                "mean_amplitude_db": float(np.nanmean(amps)),
                "peak_ping": best_ping,
                "peak_sample": best_sample,
                "peak_beam": beam,
            }
        )

    records.sort(key=lambda row: row["peak_score_mad"], reverse=True)
    for idx, rec in enumerate(records[:max_regions], start=1):
        beam = int(np.clip(rec["peak_beam"], 1, cache.n_beams))
        confidence = min(0.99, max(0.05, 0.35 + 0.08 * (rec["peak_score_mad"] - threshold_k) + 0.015 * rec["ping_span"]))
        regions.append(
            {
                "product_id": product_id,
                "region_id": f"CAND_ECHO_{idx:04d}",
                "source": "python_raw_wcd_candidate_detection",
                "ping": rec["peak_ping"],
                "ping_start": int(rec["pings"].min()),
                "ping_end": int(rec["pings"].max()),
                "beam": beam,
                "beam_start": int(np.clip(beam - 5, 1, cache.n_beams)),
                "beam_end": int(np.clip(beam + 5, 1, cache.n_beams)),
                "sample_start": int(rec["samples"].min()),
                "sample_end": int(rec["samples"].max()),
                "label": "candidate_anomalous_echo",
                "review_status": "auto_detected_unreviewed",
                "detection_method": "python_range_adaptive_median_mad_connected_components",
                "classification": "not_performed",
                "confidence": confidence,
                "area_pixels": rec["area_pixels"],
                "ping_span": rec["ping_span"],
                "sample_span": rec["sample_span"],
                "density": rec["density"],
                "peak_score_mad": rec["peak_score_mad"],
                "mean_score_mad": rec["mean_score_mad"],
                "peak_amplitude_db": rec["peak_amplitude_db"],
                "mean_amplitude_db": rec["mean_amplitude_db"],
                "bottom_clearance_samples": rec["bottom_clearance_samples"],
                "notes": "Python runtime target-candidate detection from Kongsberg water-column data.",
            }
        )

    detection = {
        "method": "python_range_adaptive_median_mad_connected_components",
        "echo_projection": "coffee_style_range_stack_linear_power_beam_mean",
        "threshold_used_k_mad": threshold_k,
        "parameters": {
            "threshold_k_mad": threshold_k,
            "bottom_guard_samples": bottom_guard,
            "max_regions": max_regions,
            "min_area_pixels": 8,
            "min_ping_span": 3,
            "min_sample_span": 2,
            "max_sample_span": 90,
        },
        "bottom_line": {
            "median_sample": float(np.nanmedian(bottom)),
            "min_sample": float(np.nanmin(bottom)),
            "max_sample": float(np.nanmax(bottom)),
            "guard_samples_above_bottom": bottom_guard,
        },
        "matrix": {"samples": n_samples, "pings": n_pings, "beams": cache.n_beams},
        "candidate_count": len(regions),
        "classification": "not_performed",
        "runtime": "python_raw_all_wcd_no_matlab",
    }
    return echo, bottom, regions, detection


def classify_rules(region: dict[str, Any], position: dict[str, Any] | None) -> tuple[str, float, str, dict[str, Any]]:
    mp = (position or {}).get("map_position", {})
    geom = (position or {}).get("observation_geometry", {})
    depth = float(mp.get("depth_m") or np.nan)
    distance = float(geom.get("distance_to_track_m") or np.nan)
    vertical_extent = float((position or {}).get("region_extent_projected_m", {}).get("height_max") or np.nan) - float((position or {}).get("region_extent_projected_m", {}).get("height_min") or np.nan)
    area = float(region.get("area_pixels") or 0)
    ping_span = float(region.get("ping_span") or 1)
    sample_span = float(region.get("sample_span") or 1)
    density = float(region.get("density") or 0)
    peak_score = float(region.get("peak_score_mad") or 0)
    peak_amp = float(region.get("peak_amplitude_db") or -999)
    features = {
        "area_pixels": area,
        "ping_span": ping_span,
        "sample_span": sample_span,
        "beam_span": max(1, int(region.get("beam_end", 0)) - int(region.get("beam_start", 0)) + 1),
        "density": density,
        "peak_score_mad": peak_score,
        "mean_score_mad": region.get("mean_score_mad"),
        "peak_amplitude_db": peak_amp,
        "mean_amplitude_db": region.get("mean_amplitude_db"),
        "bottom_clearance_samples": region.get("bottom_clearance_samples"),
        "depth_m": depth,
        "distance_to_track_m": distance,
        "vertical_extent_m": abs(vertical_extent) if math.isfinite(vertical_extent) else np.nan,
        "aspect_sample_ping": sample_span / max(1.0, ping_span),
    }
    score = min(0.93, max(0.05, 0.58 + 0.018 * min(15, max(0, peak_score - 4)) + 0.0015 * min(120, area)))
    if peak_amp > -8 or (math.isfinite(distance) and distance >= 25 and sample_span <= 25):
        return "platform_candidate", score, "strong compact return far from the vessel track", features
    if math.isfinite(depth) and depth >= 55 and sample_span >= 7 and ping_span >= 6:
        return "gas_plume_candidate", score, "multi-ping vertically extended return in deeper water", features
    if math.isfinite(depth) and 15 <= depth <= 90 and sample_span <= 25 and ping_span >= 3 and area >= 20:
        return "fish_school_candidate", score, "compact contiguous water-column blob", features
    return "unknown_target_candidate", max(0.50, score - 0.08), "valid anomalous echo candidate without a stronger rule label", features


def utm_transformer(tmproj: str) -> Transformer:
    epsg = utm_epsg_from_tmproj_or_lon(tmproj)
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)


def compute_positions(cache: RawKongsbergCache, product_id: str, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformer = utm_transformer(cache.tmproj)
    positions: list[dict[str, Any]] = []
    print("[3/6] Projecting target candidate coordinates")
    for region in regions:
        ping = int(region["ping"])
        easting, northing, height = cache.projected_coordinates(ping)
        wcd = cache.ping_wcd(ping)
        s0, s1 = int(region["sample_start"]) - 1, int(region["sample_end"])
        b0, b1 = int(region["beam_start"]) - 1, int(region["beam_end"])
        region_e = easting[s0:s1, b0:b1]
        region_n = northing[s0:s1, b0:b1]
        region_h = height[s0:s1, b0:b1]
        region_amp = wcd[s0:s1, b0:b1]
        valid = np.isfinite(region_e) & np.isfinite(region_n) & np.isfinite(region_h)
        if not valid.any():
            continue
        centroid_e = float(np.nanmean(region_e[valid]))
        centroid_n = float(np.nanmean(region_n[valid]))
        centroid_h = float(np.nanmean(region_h[valid]))
        lon, lat = transformer.transform(centroid_e, centroid_n)
        sonar_e = float(cache.easting[ping - 1])
        sonar_n = float(cache.northing[ping - 1])
        sonar_h = float(cache.height[ping - 1])
        d_e = centroid_e - sonar_e
        d_n = centroid_n - sonar_n
        d_h = centroid_h - sonar_h
        bearing_true = math.degrees(math.atan2(d_e, d_n)) % 360
        rel = wrap180(bearing_true - float(cache.heading[ping - 1]))
        track_distance = abs(float(np.nanmean(cache.wci_coordinates(ping)[0][s0:s1, b0:b1])))
        positions.append(
            {
                "product_id": product_id,
                "region_id": region["region_id"],
                "source_region": region,
                "ping_time": "",
                "map_position": {
                    "easting_m": centroid_e,
                    "northing_m": centroid_n,
                    "height_m": centroid_h,
                    "latitude_deg": lat,
                    "longitude_deg": lon,
                    "depth_m": max(0.0, -centroid_h),
                },
                "region_extent_projected_m": {
                    "easting_min": float(np.nanmin(region_e[valid])),
                    "easting_max": float(np.nanmax(region_e[valid])),
                    "northing_min": float(np.nanmin(region_n[valid])),
                    "northing_max": float(np.nanmax(region_n[valid])),
                    "height_min": float(np.nanmin(region_h[valid])),
                    "height_max": float(np.nanmax(region_h[valid])),
                },
                "observation_geometry": {
                    "sonar_easting_m": sonar_e,
                    "sonar_northing_m": sonar_n,
                    "sonar_height_m": sonar_h,
                    "slant_range_m": math.sqrt(d_e * d_e + d_n * d_n + d_h * d_h),
                    "horizontal_range_m": math.hypot(d_e, d_n),
                    "bearing_true_north_deg": bearing_true,
                    "vessel_heading_deg": float(cache.heading[ping - 1]),
                    "bearing_relative_to_heading_deg": rel,
                    "side_of_track": "right" if rel > 0 else "left" if rel < 0 else "center",
                    "distance_to_track_m": track_distance,
                    "nearest_track_ping": ping,
                },
                "signal_summary": {
                    "amplitude_mean_db": float(np.nanmean(region_amp)),
                    "amplitude_min_db": float(np.nanmin(region_amp)),
                    "amplitude_max_db": float(np.nanmax(region_amp)),
                    "sample_count": int(region_amp.size),
                },
                "quality": {
                    "status": "candidate_anomalous_echo_unreviewed",
                    "automatic_classification": False,
                    "visual_target_confirmed": False,
                    "detection_method": region.get("detection_method"),
                    "note": "Position is computed from navigation, attitude, and water-column beam geometry.",
                },
            }
        )
    return positions


def normalize_for_color(values: np.ndarray, color_axis: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    lo, hi = color_axis
    scaled = (values - lo) / max(1e-6, hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled, valid


def colorize(values: np.ndarray, color_axis: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    scaled, valid = normalize_for_color(values, color_axis)
    x = scaled
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    rgb = np.dstack([r, g, b])
    rgb[~valid] = 0
    return (rgb * 255).astype(np.uint8), (valid.astype(np.uint8) * 255)


def write_raster_png(values: np.ndarray, path: Path, color_axis: tuple[float, float] = (-50.0, -10.0), invalid_background: str = "white") -> None:
    rgb, alpha = colorize(values, color_axis)
    if invalid_background == "transparent":
        rgba = np.dstack([rgb, alpha])
    elif invalid_background == "black":
        rgba = np.dstack([rgb, np.full(alpha.shape, 255, dtype=np.uint8)])
    else:
        rgb[alpha == 0] = 255
        rgba = np.dstack([rgb, np.full(alpha.shape, 255, dtype=np.uint8)])
    Image.fromarray(rgba, mode="RGBA").save(path)


def write_overlay_png(values: np.ndarray, path: Path, x_values: np.ndarray, y_values: np.ndarray, regions: list[dict[str, Any]], color_axis: tuple[float, float] = (-50.0, -10.0)) -> None:
    write_echogram_figure(
        values,
        path,
        x_values,
        y_values,
        regions=regions,
        color_axis=color_axis,
        title="Stacked Water Column with candidate anomalous echoes",
    )


def write_echogram_figure(
    values: np.ndarray,
    path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    regions: list[dict[str, Any]] | None = None,
    color_axis: tuple[float, float] = (-50.0, -10.0),
    title: str = "Stacked Water Column",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.0, 4.8), dpi=140)
    masked = np.ma.masked_invalid(values)
    x_min, x_max = float(x_values[0]), float(x_values[-1])
    y_min, y_max = float(y_values[0]), float(y_values[-1])
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    im = ax.imshow(
        masked,
        extent=[x_min, x_max, y_max, y_min],
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=color_axis[0],
        vmax=color_axis[1],
    )
    if regions:
        for region in regions:
            r0 = sample_to_range(cache_range_axis=y_values, sample=int(region["sample_start"]))
            r1 = sample_to_range(cache_range_axis=y_values, sample=int(region["sample_end"]))
            if r0 > y_max and r1 > y_max:
                continue
            box_y0 = max(y_min, min(r0, r1))
            box_y1 = min(y_max, max(r0, r1))
            rect = Rectangle(
                (float(region["ping_start"]), box_y0),
                max(1.0, float(region["ping_end"]) - float(region["ping_start"]) + 1.0),
                max(0.25, box_y1 - box_y0),
                fill=False,
                edgecolor=(1.0, 0.45, 0.0),
                linewidth=1.4,
            )
            ax.add_patch(rect)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Ping index")
    ax.set_ylabel("Range from sonar (m)")
    ax.grid(True, color="0.45", alpha=0.25, linewidth=0.5)
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cbar.set_label("Water-column amplitude (dB)")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def write_single_ping_fan_png(cache: RawKongsbergCache, path: Path, region: dict[str, Any], color_axis: tuple[float, float] = (-60.0, 0.0)) -> None:
    ping = int(region["ping"])
    wcd = cache.ping_wcd(ping)
    across, sample_height, _ = cache.wci_coordinates(ping)
    valid = np.isfinite(wcd) & np.isfinite(across) & np.isfinite(sample_height)
    if not valid.any():
        write_raster_png(wcd, path, color_axis)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    masked = np.ma.masked_invalid(wcd)
    try:
        mesh = ax.pcolormesh(
            across,
            sample_height,
            masked,
            shading="auto",
            cmap=cmap,
            vmin=color_axis[0],
            vmax=color_axis[1],
        )
    except Exception:
        mesh = ax.scatter(
            across[valid],
            sample_height[valid],
            c=wcd[valid],
            s=0.8,
            linewidths=0,
            cmap=cmap,
            vmin=color_axis[0],
            vmax=color_axis[1],
        )
    x_abs = float(np.nanmax(np.abs(across[valid])))
    y_min = float(np.nanmin(sample_height[valid]))
    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(y_min, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Single ping WCI fan view: ping {ping}", fontsize=10)
    ax.set_xlabel("Across-track distance from sonar (m)")
    ax.set_ylabel("Height above sonar (m)")
    ax.grid(True, color="0.45", alpha=0.25, linewidth=0.5)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.04)
    cbar.set_label("Water-column amplitude (dB)")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def sample_to_range(cache_range_axis: np.ndarray, sample: int) -> float:
    idx = int(np.clip(sample - 1, 0, len(cache_range_axis) - 1))
    return float(cache_range_axis[idx])


def export_render_data(output_dir: Path, echo: np.ndarray, bottom: np.ndarray, range_axis: np.ndarray, regions: list[dict[str, Any]]) -> None:
    pings = np.arange(1, echo.shape[1] + 1, dtype=int)
    masked = echo.astype(float).copy()
    bottom_range = np.array([sample_to_range(range_axis, int(x)) if np.isfinite(x) else np.nan for x in bottom])
    finite_bottom = bottom_range[np.isfinite(bottom_range)]
    if finite_bottom.size:
        median_bottom = float(np.nanmedian(finite_bottom))
        y_max = min(float(range_axis[-1]), max(20.0, float(np.nanpercentile(finite_bottom + max(16.0, median_bottom * 0.22), 98))))
    else:
        y_max = float(range_axis[-1])
    for col, b in enumerate(bottom):
        if np.isfinite(b):
            cut = min(masked.shape[0], int(round(float(b))) + 4)
            masked[cut:, col] = np.nan
    keep = range_axis <= y_max
    render = masked[keep, :]
    render_ranges = range_axis[keep]
    write_echogram_figure(render, output_dir / "echogram_raster.png", pings, render_ranges, None, (-50.0, -10.0), "Stacked Water Column")
    write_echogram_figure(render, output_dir / "echogram.png", pings, render_ranges, None, (-50.0, -10.0), "Stacked Water Column")
    write_overlay_png(render, output_dir / "candidate_detection_overlay.png", pings, render_ranges, regions, (-50.0, -10.0))
    write_overlay_png(render, output_dir / "candidate_classification_overlay.png", pings, render_ranges, regions, (-50.0, -10.0))

    rounded = np.where(np.isfinite(render), np.round(render * 100) / 100, np.nan)
    write_json(
        output_dir / "echogram_render_data.json",
        {
            "schema": "active-sonar-echogram-render-data.v1",
            "rendering": "python_raw_wcd_range_stack_matrix",
            "x_axis": {"field": "ping", "min": int(pings[0]), "max": int(pings[-1]), "values": pings.tolist()},
            "y_axis": {"field": "range_m", "min": float(render_ranges[0]), "max": float(render_ranges[-1]), "values": np.round(render_ranges, 3).tolist()},
            "color_axis_db": [-50, -10],
            "value_units": "dB",
            "null_value": "masked_or_invalid_sample",
            "matrix_orientation": "matrix[row][col] maps to y_axis.values[row], x_axis.values[col]",
            "matrix": rounded.tolist(),
        },
    )
    write_json(
        output_dir / "echogram_render_meta.json",
        {
            "product_asset": "echogram_raster.png",
            "rendering": "rgba_data_layer_only_no_matlab_axes",
            "x_axis": {"field": "ping", "min": int(pings[0]), "max": int(pings[-1]), "values": pings.tolist()},
            "y_axis": {"field": "range_m", "min": float(render_ranges[0]), "max": float(render_ranges[-1]), "values": np.round(render_ranges, 3).tolist()},
            "color_axis_db": [-50, -10],
            "transparent_value": "invalid_or_below_bottom_mask",
            "image_width_px": int(render.shape[1]),
            "image_height_px": int(render.shape[0]),
        },
    )
    overlay_regions = []
    for region in regions:
        overlay_regions.append(
            {
                "product_id": region["product_id"],
                "region_id": region["region_id"],
                "ping_start": region["ping_start"],
                "ping_end": region["ping_end"],
                "ping": region["ping"],
                "beam_start": region["beam_start"],
                "beam_end": region["beam_end"],
                "sample_start": region["sample_start"],
                "sample_end": region["sample_end"],
                "range_start_m": sample_to_range(range_axis, int(region["sample_start"])),
                "range_end_m": sample_to_range(range_axis, int(region["sample_end"])),
                "label": region["region_id"],
                "classification": region.get("classification", "candidate_anomalous_echo"),
                "confidence": region.get("classification_confidence", region.get("confidence")),
            }
        )
    write_json(
        output_dir / "candidate_overlay.json",
        {"schema": "active-sonar-echogram-overlay.v1", "coordinate_space": {"x": "ping", "y": "range_m"}, "regions": overlay_regions},
    )


def export_wci_products(cache: RawKongsbergCache, output_dir: Path, regions: list[dict[str, Any]]) -> None:
    crop_dir = output_dir / "candidate_crops"
    wci_render_dir = output_dir / "candidate_wci_render_data"
    crop_dir.mkdir(parents=True, exist_ok=True)
    wci_render_dir.mkdir(parents=True, exist_ok=True)
    crops: list[dict[str, Any]] = []
    focus = regions[0] if regions else {"region_id": "", "ping": max(1, cache.n_pings // 2), "sample_start": 1, "sample_end": 2, "beam_start": 1, "beam_end": 2}

    print("[5/6] Exporting WCI render data and model input crops")
    for region in regions:
        ping = int(region["ping"])
        wcd = cache.ping_wcd(ping)
        s0 = max(0, int(region["sample_start"]) - 35)
        s1 = min(cache.n_samples, int(region["sample_end"]) + 35)
        b0 = max(0, int(region["beam_start"]) - 65)
        b1 = min(cache.n_beams, int(region["beam_end"]) + 65)
        crop = wcd[s0:s1, b0:b1]
        crop_path = crop_dir / f"{region['region_id']}_crop.png"
        wci_crop_path = crop_dir / f"{region['region_id']}_wci_crop.png"
        preview_path = crop_dir / f"{region['region_id']}_preview.png"
        wci_render_path = wci_render_dir / f"{region['region_id']}.json"
        write_raster_png(crop, crop_path, (-60, 0))
        shutil.copyfile(crop_path, wci_crop_path)
        shutil.copyfile(crop_path, preview_path)
        export_single_ping_render(cache, wci_render_path, region)
        crops.append(
            {
                "product_id": region["product_id"],
                "region_id": region["region_id"],
                "crop_png": str(crop_path),
                "wci_crop_png": str(wci_crop_path),
                "ml_input_png": str(wci_crop_path),
                "wci_render_data_json": str(wci_render_path),
                "preview_png": str(preview_path),
                "wci_preview_png": str(preview_path),
                "source_echogram_png": str(output_dir / "echogram.png"),
                "ping_start": region["ping_start"],
                "ping_end": region["ping_end"],
                "range_start_m": cache.inter_sample_distance(ping) * region["sample_start"],
                "range_end_m": cache.inter_sample_distance(ping) * region["sample_end"],
                "candidate_sample_start": region["sample_start"],
                "candidate_sample_end": region["sample_end"],
                "candidate_beam_start": region["beam_start"],
                "candidate_beam_end": region["beam_end"],
                "rule_classification": region.get("classification", "not_performed"),
                "rule_confidence": region.get("classification_confidence", region.get("confidence")),
                "latitude_deg": "",
                "longitude_deg": "",
                "depth_m": "",
                "ml_status": "pending_yolo_inference",
            }
    )

    export_single_ping_render(cache, output_dir / "single_ping_render_data.json", focus)
    write_single_ping_fan_png(cache, output_dir / "single_ping.png", focus, (-60, 0))
    write_json(
        output_dir / "candidate_crop_manifest.json",
        {
            "schema": {
                "product_id": "string",
                "region_id": "string",
                "crop_png": "string",
                "wci_crop_png": "string",
                "ml_input_png": "string",
            },
            "note": "Python generated candidate crops from raw Kongsberg WCI data.",
            "crops": crops,
        },
    )
    write_csv(
        output_dir / "candidate_crop_manifest.csv",
        crops,
        [
            "product_id",
            "region_id",
            "crop_png",
                "wci_crop_png",
                "ml_input_png",
                "wci_render_data_json",
                "preview_png",
            "ping_start",
            "ping_end",
            "range_start_m",
            "range_end_m",
            "candidate_sample_start",
            "candidate_sample_end",
            "candidate_beam_start",
            "candidate_beam_end",
            "rule_classification",
            "rule_confidence",
            "latitude_deg",
            "longitude_deg",
            "depth_m",
            "ml_status",
        ],
    )


def export_single_ping_render(cache: RawKongsbergCache, out_file: Path, region: dict[str, Any]) -> None:
    ping = int(region["ping"])
    wcd = cache.ping_wcd(ping)
    across, height, _ = cache.wci_coordinates(ping)
    n_samples, n_beams = wcd.shape
    max_points = 80000
    step = max(1, int(math.ceil(math.sqrt(wcd.size / max_points))))
    sample_idx = np.arange(0, n_samples, step)
    beam_idx = np.arange(0, n_beams, step)
    sub_amp = wcd[np.ix_(sample_idx, beam_idx)]
    sub_x = across[np.ix_(sample_idx, beam_idx)]
    sub_y = height[np.ix_(sample_idx, beam_idx)]
    ss, bb = np.meshgrid(sample_idx + 1, beam_idx + 1, indexing="ij")
    valid = np.isfinite(sub_amp) & np.isfinite(sub_x) & np.isfinite(sub_y)
    points = np.column_stack([sub_x[valid], sub_y[valid], sub_amp[valid], ss[valid], bb[valid]])
    write_json(
        out_file,
        {
            "schema": "active-sonar-single-ping-wci-render-data.v1",
            "rendering": "python_single_ping_wci_points",
            "ping": ping,
            "color_axis_db": [-60, 0],
            "value_units": "dB",
            "coordinate_space": {
                "x": "across_track_distance_from_sonar_m",
                "y": "height_above_sonar_m",
                "depth": "depth_from_sonar_m",
            },
            "source_dimensions": {"samples": n_samples, "beams": n_beams},
            "display_decimation": {"sample_step": step, "beam_step": step, "point_count": int(points.shape[0])},
            "x_axis": {"field": "across_track_distance_from_sonar_m", "min": float(np.nanmin(points[:, 0])), "max": float(np.nanmax(points[:, 0]))},
            "y_axis": {"field": "height_above_sonar_m", "min": float(np.nanmin(points[:, 1])), "max": float(np.nanmax(points[:, 1]))},
            "columns": ["across_track_m", "height_above_sonar_m", "amplitude_db", "sample", "beam"],
            "points": np.round(points, 3).tolist(),
            "focus_region": {
                "region_id": region.get("region_id", ""),
                "ping": ping,
                "sample_start": region.get("sample_start"),
                "sample_end": region.get("sample_end"),
                "beam_start": region.get("beam_start"),
                "beam_end": region.get("beam_end"),
            },
        },
    )


def write_product_files(
    cache: RawKongsbergCache,
    output_dir: Path,
    product_id: str,
    survey_id: str,
    all_file: Path,
    wcd_file: Path,
    regions: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    detection: dict[str, Any],
) -> None:
    transformer = utm_transformer(cache.tmproj)
    lons, lats = transformer.transform(cache.easting, cache.northing)
    product = {
        "product_id": product_id,
        "product_type": "active_multibeam_water_column",
        "survey_id": survey_id,
        "platform": "TecnopescaII",
        "sonar_model": "Kongsberg EM2040CD",
        "frequency_hz": 300000,
        "raw_files": {"all": str(all_file), "wcd": str(wcd_file)},
        "processing": {
            "engine": "Python raw Kongsberg .all/.wcd reader",
            "runtime": "python_no_matlab",
            "conversion_type": "raw_Kongsberg_ALL_WCD",
            "tm_projection": cache.tmproj,
            "ellipsoid": cache.ellips,
        },
        "dimensions": {"samples": cache.n_samples, "beams": cache.n_beams, "pings": cache.n_pings},
        "track": {
            "point_count": cache.n_pings,
            "easting_min_m": float(np.nanmin(cache.easting)),
            "easting_max_m": float(np.nanmax(cache.easting)),
            "northing_min_m": float(np.nanmin(cache.northing)),
            "northing_max_m": float(np.nanmax(cache.northing)),
            "latitude_min_deg": float(np.nanmin(lats)),
            "latitude_max_deg": float(np.nanmax(lats)),
            "longitude_min_deg": float(np.nanmin(lons)),
            "longitude_max_deg": float(np.nanmax(lons)),
        },
    }
    write_json(output_dir / "product.json", product)

    track_rows = []
    for i in range(cache.n_pings):
        track_rows.append(
            {
                "ping": i + 1,
                "ping_counter": cache.ping_counter[i],
                "ping_time": "",
                "easting_m": cache.easting[i],
                "northing_m": cache.northing[i],
                "height_m": cache.height[i],
                "heading_deg": cache.heading[i],
                "speed_mps": cache.speed[i] if i < cache.speed.size else "",
                "latitude_deg": lats[i],
                "longitude_deg": lons[i],
            }
        )
    write_csv(output_dir / "track_points.csv", track_rows, list(track_rows[0].keys()))

    print("[4/6] Writing candidate regions, positions and rule labels")
    class_rows = []
    pos_by_id = {row["region_id"]: row for row in positions}
    for region in regions:
        label, conf, reason, features = classify_rules(region, pos_by_id.get(region["region_id"]))
        region["classification"] = label
        region["classification_method"] = "python_rule_based_water_column_candidate_classifier_v1"
        region["classification_confidence"] = conf
        region["classification_reason"] = reason
        class_rows.append(
            {
                "product_id": product_id,
                "region_id": region["region_id"],
                "candidate_type": label,
                "classification": label,
                "confidence": conf,
                "method": region["classification_method"],
                "reason": reason,
                "features": features,
            }
        )

    region_payload = {
        "product_id": product_id,
        "schema": {},
        "detection": {**detection, "classification": "rule_based_candidate_type_assigned"},
        "regions": regions,
    }
    position_payload = {
        "product_id": product_id,
        "coordinate_frame": {
            "projected": cache.tmproj,
            "ellipsoid": cache.ellips,
            "height_positive": "up",
            "depth_m": "computed as -height_m for map display",
        },
        "positions": positions,
        "classification": "rule_based_candidate_type_assigned",
        "note": "Python generated candidate positions from raw Kongsberg navigation, attitude, and beam geometry.",
    }
    classification_payload = {
        "product_id": product_id,
        "method": "python_rule_based_water_column_candidate_classifier_v1",
        "classifications": class_rows,
        "warning": "Rule labels are fallback candidate types. Final target class comes from the YOLO model when it fires.",
    }
    write_json(output_dir / "candidate_regions.json", region_payload)
    write_json(output_dir / "target_regions.json", {**region_payload, "alias_of": "candidate_regions.json"})
    write_json(output_dir / "candidate_positions.json", position_payload)
    write_json(output_dir / "target_positions.json", {**position_payload, "alias_of": "candidate_positions.json"})
    write_json(output_dir / "candidate_classifications.json", classification_payload)

    region_fields = [
        "product_id",
        "region_id",
        "source",
        "ping",
        "ping_start",
        "ping_end",
        "beam",
        "beam_start",
        "beam_end",
        "sample_start",
        "sample_end",
        "label",
        "review_status",
        "detection_method",
        "classification",
        "classification_confidence",
        "confidence",
        "area_pixels",
        "ping_span",
        "sample_span",
        "density",
        "peak_score_mad",
        "mean_score_mad",
        "peak_amplitude_db",
        "mean_amplitude_db",
        "bottom_clearance_samples",
    ]
    write_csv(output_dir / "candidate_regions.csv", regions, region_fields)
    shutil.copyfile(output_dir / "candidate_regions.csv", output_dir / "target_regions.csv")

    pos_rows = []
    for pos in positions:
        mp = pos["map_position"]
        geom = pos["observation_geometry"]
        src = pos["source_region"]
        pos_rows.append(
            {
                "product_id": product_id,
                "region_id": pos["region_id"],
                "ping": src["ping"],
                "beam": src["beam"],
                "beam_start": src["beam_start"],
                "beam_end": src["beam_end"],
                "sample_start": src["sample_start"],
                "sample_end": src["sample_end"],
                "easting_m": mp["easting_m"],
                "northing_m": mp["northing_m"],
                "height_m": mp["height_m"],
                "latitude_deg": mp["latitude_deg"],
                "longitude_deg": mp["longitude_deg"],
                "depth_m": mp["depth_m"],
                "slant_range_m": geom["slant_range_m"],
                "horizontal_range_m": geom["horizontal_range_m"],
                "distance_to_track_m": geom["distance_to_track_m"],
                "bearing_true_north_deg": geom["bearing_true_north_deg"],
                "bearing_relative_to_heading_deg": geom["bearing_relative_to_heading_deg"],
                "side_of_track": geom["side_of_track"],
            }
        )
    write_csv(output_dir / "candidate_positions.csv", pos_rows, list(pos_rows[0].keys()) if pos_rows else ["product_id", "region_id"])
    shutil.copyfile(output_dir / "candidate_positions.csv", output_dir / "target_positions.csv")

    flat_class_rows = []
    for row in class_rows:
        flat = {k: v for k, v in row.items() if k != "features"}
        flat.update(row["features"])
        flat_class_rows.append(flat)
    write_csv(output_dir / "candidate_classifications.csv", flat_class_rows, list(flat_class_rows[0].keys()) if flat_class_rows else ["product_id", "region_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure Python active-sonar product generator from raw Kongsberg .all/.wcd data.")
    parser.add_argument("--survey-id", default="20181112_survey")
    parser.add_argument("--all-name", default="0000_20181112_080147_TecnopescaII.all")
    parser.add_argument("--product-id", default="")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--threshold-k-mad", type=float, default=3.8)
    parser.add_argument("--bottom-guard-samples", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=80)
    args = parser.parse_args()

    all_file = locate_all_file(args.data_root, args.survey_id, args.all_name)
    wcd_file = all_file.with_suffix(".wcd")
    product_id = args.product_id or product_id_from_all_name(args.survey_id, all_file.name)
    output_dir = args.output_root / product_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Active sonar Python raw-data pipeline")
    print(f"Root:       {ROOT}")
    print(f"Survey:     {args.survey_id}")
    print(f"ALL file:   {all_file}")
    print(f"WCD file:   {wcd_file}")
    print(f"Product ID: {product_id}")
    print(f"Output:     {output_dir}")
    print("Data mode:   raw Kongsberg .all/.wcd")
    cache = RawKongsbergCache(all_file, wcd_file)

    try:
        echo, bottom, regions, detection = build_detection(cache, args.threshold_k_mad, args.bottom_guard_samples, args.max_regions, product_id)
        positions = compute_positions(cache, product_id, regions)
        write_product_files(cache, output_dir, product_id, args.survey_id, all_file, wcd_file, regions, positions, detection)
        range_axis = cache.range_axis()
        export_render_data(output_dir, echo, bottom, range_axis, regions)
        export_wci_products(cache, output_dir, regions)
        print("[6/6] Python product generation complete")
        print(output_dir / "product.json")
        print(output_dir / "candidate_regions.json")
        print(output_dir / "candidate_positions.json")
        print(output_dir / "candidate_crop_manifest.json")
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
