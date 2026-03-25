from pathlib import Path


def test_geo_placeholder_csv_files_exist_with_expected_headers() -> None:
    expected_headers = {
        Path("config/geo/asset_region_exposure.csv"): "asset,region_code,weight",
        Path("config/geo/asset_sector_exposure.csv"): "asset,sector_code,weight",
        Path("config/geo/asset_infra_exposure.csv"): "asset,infra_code,weight",
        Path("config/geo/event_betas.csv"): "event_type,event_subtype,dimension_type,dimension_key,beta",
    }

    for path, header in expected_headers.items():
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip() == header
