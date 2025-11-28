#!/usr/bin/env python3
"""
download_nasdaq_tickers.py

Скачивает список тикеров NASDAQ (nasdaqlisted.txt) с FTP NASDAQ
и сохраняет их в простой TXT-файл: один тикер в строке.
"""

from ftplib import FTP
from pathlib import Path
import io


def fetch_nasdaq_listed_symbols():
    """
    Забираем файл nasdaqlisted.txt с FTP и парсим список тикеров.
    Файл: ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt
    """
    ftp = FTP("ftp.nasdaqtrader.com")
    ftp.login()  # anonymous
    ftp.cwd("SymbolDirectory")

    buf = io.BytesIO()
    ftp.retrbinary("RETR nasdaqlisted.txt", buf.write)
    ftp.quit()

    buf.seek(0)
    text = buf.read().decode("utf-8").splitlines()
    if not text:
        return []

    header = text[0].split("|")
    rows = text[1:-1]  # последняя строка: "File Creation Time: ..."

    def parse_row(line):
        parts = line.split("|")
        return dict(zip(header, parts))

    symbols = []
    for line in rows:
        row = parse_row(line)
        sym = (row.get("Symbol") or "").strip().upper()
        test_issue = (row.get("Test Issue") or "").strip().upper()
        fin_status = (row.get("Financial Status") or "").strip().upper()

        if not sym:
            continue
        # Отбрасываем тестовые и проблемные бумаги
        if test_issue == "Y":
            continue
        if fin_status in {"D", "E"}:  # deficient / delinquent
            continue
        # Часто "спец"-тикеры с . или $ — можно выкинуть
        if any(ch in sym for ch in (".", "$")):
            continue

        symbols.append(sym)

    # Убираем дубликаты и сортируем
    unique_sorted = sorted(set(symbols))
    return unique_sorted


def save_tickers_to_txt(tickers, outfile: Path):
    outfile = Path(outfile)
    outfile.write_text("\n".join(tickers), encoding="utf-8")
    print(f"Сохранил {len(tickers)} тикеров в {outfile.resolve()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Скачать тикеры NASDAQ и сохранить в TXT (один тикер в строке)."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="nasdaq_tickers.txt",
        help="путь к выходному txt-файлу (по умолчанию nasdaq_tickers.txt)",
    )
    args = parser.parse_args()

    tickers = fetch_nasdaq_listed_symbols()
    if not tickers:
        raise SystemExit("Не удалось получить ни одного тикера c NASDAQ FTP")

    save_tickers_to_txt(tickers, Path(args.output))


if __name__ == "__main__":
    main()
