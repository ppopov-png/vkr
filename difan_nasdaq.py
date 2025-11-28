# -*- coding: utf-8 -*-

"""

difan_nasdaq.py



1) Логинится на difan.

2) Открывает страницу HSE.

3) Выбирает биржу XNAS (NASDAQ).

4) Ждёт, пока в списке компаний появятся NASDAQ-тикеры (например, AAPL).

5) Собирает все тикеры для выбранной биржи.

6) Опционально выкидывает тикеры, оканчивающиеся на .SA.

7) Сохраняет тикеры в txt (TICKERS_TXT_PATH).

8) По каждому тикеру скачивает все доступные отчёты в downloads/<ticker>/.

"""



import os
import time
import re
import datetime as dt
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager





# ================== ХЕЛПЕРЫ ДЛЯ ТИКЕРОВ ==================



def clean_ticker_raw(value: str) -> str:

    if not value:

        return ""

    return re.sub(r"\(.*?\)", "", value).strip()





def base_ticker_symbol(ticker: str) -> str:

    if not ticker:

        return ""

    cleaned = clean_ticker_raw(str(ticker))

    return cleaned.split("-", 1)[0].strip().upper()





def normalize_tickers(seq):

    seen = set()

    out = []

    for item in seq or []:

        raw = clean_ticker_raw(str(item or "")).upper()

        if not raw:

            continue

        val = base_ticker_symbol(raw)

        if not val or val in seen:

            continue

        seen.add(val)

        out.append(val)

    return out





def safe_name(s):

    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)





def normalize_report_label(label: str) -> str:

    """

    Normalize report labels so we can compare them regardless of spacing/case.

    """

    if not label:

        return ""

    return re.sub(r"\s+", " ", label).strip().lower()





def ticker_folder(root_dir: Path, ticker: str) -> Path:

    base = base_ticker_symbol(ticker)

    if not base:

        base = ticker.upper()

    folder = root_dir / safe_name(base)

    folder.mkdir(parents=True, exist_ok=True)

    return folder





# ================== КОНФИГ ==================



PAGE_URL = "https://difan.xyz/HSE/"

LOGIN_URL = os.environ.get("DIFAN_LOGIN_URL", "https://difan.xyz/en")

LOGIN_EMAIL = os.environ.get("DIFAN_LOGIN", "difanxyz@gmail.com")

LOGIN_PASSWORD = os.environ.get("DIFAN_PASSWORD", "student")



DOWNLOAD_ROOT = Path.cwd() / "downloads"
DEFAULT_DOWNLOAD_FALLBACK = Path.home() / "Downloads"
HEADLESS = False
WAIT_TIMEOUT = 5  # seconds
EXCHANGE_CODE = "XNAS"  # NASDAQ



TICKERS_TXT_PATH = Path(

    r"C:\Users\Павел\Desktop\Диплом финансы\nasdaq_tickers.txt"

)




DESIRED_REPORT_NAMES = [

    "Annual balance sheet statements",

    "Annual income statements",

    "Annual cash flow statements",

    "Annual enterprise values",

    "Quarterly balance sheet statement",

    "Quarterly income statements",

    "Quarterly cash flow statements",

    "Quarterly enterprise values",

    "Historical Daily Prices",

    "Historical Monthly Prices",

    "Company dividends",

    "ESG Score",

    "ESG Risk Ratings",

]



DESIRED_REPORT_NAMES_NORMALIZED = {

    normalize_report_label(name): name for name in DESIRED_REPORT_NAMES

}

DIVIDENDS_REPORT_LABEL = "Company dividends"
DIVIDENDS_REPORT_NORMALIZED = normalize_report_label(DIVIDENDS_REPORT_LABEL)

MAX_DIVIDEND_TICKERS = 1000


# ================== BROWSER / SELENIUM ==================



def make_driver(download_dir: Path, headless: bool = False):

    download_dir = Path(download_dir).resolve()

    download_dir.mkdir(parents=True, exist_ok=True)



    options = webdriver.ChromeOptions()

    options.add_argument("--disable-blink-features=AutomationControlled")

    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    options.add_experimental_option("useAutomationExtension", False)

    options.add_argument("--ignore-certificate-errors")

    options.add_argument("--no-sandbox")

    options.add_argument("--disable-gpu")

    options.add_argument("--start-maximized")



    prefs = {

        "download.default_directory": str(download_dir),

        "download.prompt_for_download": False,

        "download.directory_upgrade": True,

        "safebrowsing.enabled": True,

        "profile.default_content_setting_values.automatic_downloads": 1,

        "profile.content_settings.exceptions.automatic_downloads.*.setting": 1,

    }

    options.add_experimental_option("prefs", prefs)



    if headless:

        options.add_argument("--headless=new")



    driver = webdriver.Chrome(

        service=Service(ChromeDriverManager().install()),

        options=options,

    )



    try:

        driver.execute_cdp_cmd(

            "Page.setDownloadBehavior",

            {"behavior": "allow", "downloadPath": str(download_dir)},

        )

    except Exception:

        pass



    return driver





def set_download_destination(driver, folder: Path):

    folder = Path(folder).resolve()

    folder.mkdir(parents=True, exist_ok=True)

    try:

        driver.execute_cdp_cmd(

            "Page.setDownloadBehavior",

            {"behavior": "allow", "downloadPath": str(folder)},

        )

    except Exception as e:

        print(f"[download-dir] cannot set download dir: {e}")





# ================== ЛОГИН / НАВИГАЦИЯ ==================



def perform_login(driver, email: str, password: str):

    if not email or not password:

        print("[login] no credentials, skipping login")

        return



    print("[login] opening login page...")

    driver.get(LOGIN_URL)



    try:

        login_input = WebDriverWait(driver, WAIT_TIMEOUT).until(

            EC.element_to_be_clickable((By.ID, "login"))

        )

        login_input.clear()

        login_input.send_keys(email)



        pwd_input = WebDriverWait(driver, WAIT_TIMEOUT).until(

            EC.element_to_be_clickable((By.ID, "password"))

        )

        pwd_input.clear()

        pwd_input.send_keys(password)



        btn = WebDriverWait(driver, WAIT_TIMEOUT).until(

            EC.element_to_be_clickable((By.ID, "auth"))

        )

        btn.click()



        def _logged_in(drv):

            url = drv.current_url.lower()

            if "hse" in url or "dataset" in url:

                return True

            try:

                drv.find_element(By.LINK_TEXT, "Dataset")

                return True

            except Exception:

                return False



        WebDriverWait(driver, WAIT_TIMEOUT).until(_logged_in)

        print("[login] success")

    except Exception as e:

        print(f"[login] failed: {e}")





def open_hse_page(driver):

    print("[nav] opening HSE page...")

    driver.get(PAGE_URL)

    WebDriverWait(driver, WAIT_TIMEOUT).until(

        EC.presence_of_element_located((By.ID, "exchangeList"))

    )

    WebDriverWait(driver, WAIT_TIMEOUT).until(

        EC.presence_of_element_located((By.ID, "requestList"))

    )

    print("[nav] HSE page opened")





# ================== ВЫБОР БИРЖИ / ТИКЕРОВ / ОТЧЁТОВ ==================



def pick_exchange(driver, code: str):

    code = (code or "").strip().upper()

    if not code:

        raise RuntimeError("Empty exchange code")



    sel = WebDriverWait(driver, WAIT_TIMEOUT).until(

        EC.element_to_be_clickable((By.ID, "exchangeList"))

    )

    s = Select(sel)



    target = None

    for opt in s.options:

        value = (opt.get_attribute("value") or "").strip().upper()

        if value == code:

            target = opt

            break



    if not target:

        for opt in s.options:

            text = (opt.text or "").strip().lower()

            if "nasdaq" in text:

                target = opt

                break



    if not target:

        raise RuntimeError(f"Exchange '{code}' not found on the page")



    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", target)

    target.click()

    print(f"[exchange] selected: {target.text.strip()} ({code})")





def wait_company_list_for_nasdaq(driver, timeout: int = WAIT_TIMEOUT):

    """

    Ждём, пока список компаний будет явно для NASDAQ.

    Ориентир — появление AAPL, MSFT или AMZN в listCompany / datalist.

    """

    print("[tickers] waiting for NASDAQ tickers (e.g. AAPL/MSFT/AMZN)...")



    def _has_nasdaq_samples(drv):

        script = """

            const vals = [];

            const list = (window.listCompany || []);

            for (const x of list) {

                vals.push(String(x || '').toUpperCase());

            }

            const opts = document.querySelectorAll('#companyList option');

            for (const o of opts) {

                vals.push(String(o.value || '').toUpperCase());

            }

            const samples = ['AAPL','MSFT','AMZN'];

            return vals.some(v => samples.includes(v) || samples.some(s => v.startsWith(s)));

        """

        try:

            return bool(drv.execute_script(script))

        except Exception:

            return False



    WebDriverWait(driver, timeout).until(_has_nasdaq_samples)

    print("[tickers] NASDAQ tickers detected")





def collect_tickers_from_page(driver):

    """

    Берём тикеры из window.listCompany или datalist#companyList.

    """

    script = """

        const list = (window.listCompany || []).slice();

        if (list.length) {

            return list.map(x => String(x || ''));

        }

        const opts = document.querySelectorAll('#companyList option');

        return Array.from(opts).map(o => o.value || '').filter(Boolean);

    """

    try:

        raw = driver.execute_script(script) or []

    except Exception as e:

        print(f"[tickers] JS error: {e}")

        raw = []



    tickers = normalize_tickers(raw)

    return tickers





def save_tickers_to_txt(tickers, path: Path):

    path = Path(path)

    if not path.parent.exists():

        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("\n".join(tickers), encoding="utf-8")

    print(f"[tickers] saved {len(tickers)} tickers to {path}")





def pick_ticker(driver, ticker: str, retries: int = 3):

    """

    Вводит тикер в companyListInput, подставляет подходящее значение из datalist

    и проверяет, что реально выбранная бумага совпадает по базовому тикеру.

    Если страница подставляет другую бумагу (например, STEL для MAPT4.SA),

    выбрасываем исключение, чтобы такой тикер просто пропустить.

    """

    base = base_ticker_symbol(ticker)

    for attempt in range(1, retries + 1):

        try:

            inp = WebDriverWait(driver, WAIT_TIMEOUT).until(

                EC.element_to_be_clickable((By.ID, "companyListInput"))

            )

            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", inp)

            inp.click()

            inp.clear()

            inp.send_keys(base)



            try:

                WebDriverWait(driver, WAIT_TIMEOUT).until(

                    EC.presence_of_element_located(

                        (By.XPATH, "//datalist[@id='companyList']/option")

                    )

                )

            except TimeoutException:

                pass



            opts_script = """

                const opts = document.querySelectorAll('#companyList option');

                return Array.from(opts).map(o => o.value || '').filter(Boolean);

            """

            try:

                options = driver.execute_script(opts_script) or []

            except Exception:

                options = []



            options = [o for o in options if o]



            chosen = None

            base_upper = base.upper()



            for o in options:

                if o.upper() == base_upper:

                    chosen = o

                    break



            if not chosen:

                for o in options:

                    if o.upper().startswith(base_upper):

                        chosen = o

                        break



            if not chosen and options:

                chosen = options[0]



            if not chosen:

                raise RuntimeError(f"no matching companyList option for '{ticker}'")



            driver.execute_script(

                "arguments[0].value = arguments[1];"

                "arguments[0].dispatchEvent(new Event('input'));"

                "arguments[0].dispatchEvent(new Event('change'));",

                inp,

                chosen,

            )

            driver.execute_script("arguments[0].blur();", inp)

            time.sleep(0.5)



            actual = driver.execute_script(

                "const el = document.getElementById('companyListInput');"

                "return el ? String(el.value || '') : '';"

            ) or ""

            if base_ticker_symbol(actual) != base_ticker_symbol(ticker):

                raise RuntimeError(

                    f"UI picked different ticker '{actual}' for '{ticker}'"

                )



            return

        except StaleElementReferenceException:

            if attempt == retries:

                raise

            time.sleep(0.5)

        except TimeoutException:

            if attempt == retries:

                raise

            time.sleep(0.5)





def get_available_report_options(driver):

    """

    Возвращает (value, text) для всех включённых отчётов в #requestList.

    """

    sel = WebDriverWait(driver, WAIT_TIMEOUT).until(

        EC.presence_of_element_located((By.ID, "requestList"))

    )

    s = Select(sel)

    result = []

    for opt in s.options:

        if opt.get_attribute("disabled"):

            continue

        text = (opt.text or "").strip()

        value = (opt.get_attribute("value") or "").strip()

        if not text:

            continue

        result.append((value, text))

    return result





def filter_desired_reports(report_options):



    """



    Filter report options to only those explicitly requested by the user.



    Returns (selected_reports, missing_labels).



    """



    selected = []



    seen = set()



    for value, text in report_options:



        norm = normalize_report_label(text)



        if norm in DESIRED_REPORT_NAMES_NORMALIZED:



            selected.append((value, text))



            seen.add(norm)



    missing = [



        DESIRED_REPORT_NAMES_NORMALIZED[norm]



        for norm in DESIRED_REPORT_NAMES_NORMALIZED



        if norm not in seen



    ]



    return selected, missing



def select_report(driver, value: str, text: str):

    """
    Select report option by matching both value and visible text (needed because many options share the same value).
    """

    sel = WebDriverWait(driver, WAIT_TIMEOUT).until(

        EC.element_to_be_clickable((By.ID, "requestList"))

    )

    s = Select(sel)

    norm_target = normalize_report_label(text)

    options = []

    for idx, opt in enumerate(s.options):

        opt_value = (opt.get_attribute("value") or "").strip()

        opt_text = (opt.text or "").strip()

        opt_norm = normalize_report_label(opt_text)

        options.append((idx, opt, opt_value, opt_text, opt_norm))

    # try exact text match first
    target = next((o for o in options if o[4] == norm_target), None)

    if not target:

        # fallback: match by value + partial text
        target = next(
            (o for o in options if o[2] == value and norm_target in o[4]),
            None,
        )

    if not target:

        raise RuntimeError(f"option '{text}' (value='{value}') not found")

    target_idx, target_opt, target_val, target_text, _ = target

    attempts = 5

    last_err = None

    for _ in range(attempts):

        try:

            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", sel)

            driver.execute_script("arguments[0].click();", sel)

            driver.execute_script(

                "const sel=arguments[0]; const idx=arguments[1];"

                "if(sel && sel.options && sel.options.length>idx){"

                "sel.selectedIndex = idx;"

                "sel.value = sel.options[idx].value;"

                "sel.options[idx].selected = true;"

                "sel.dispatchEvent(new Event('input',{bubbles:true}));"

                "sel.dispatchEvent(new Event('change',{bubbles:true}));}",

                sel,

                target_idx,

            )

            time.sleep(0.2)

        except Exception as e:

            last_err = e



        time.sleep(0.2)

        try:

            current_text = s.first_selected_option.text.strip()

            if normalize_report_label(current_text) == norm_target:

                return

        except Exception:

            pass



    if last_err:

        raise last_err

    raise RuntimeError(f"cannot select report '{text}' (value='{value}')")





def click_get_data(driver):

    btn = WebDriverWait(driver, WAIT_TIMEOUT).until(

        EC.element_to_be_clickable((By.ID, "get_data"))

    )

    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)

    btn.click()





# ================== ОТСЛЕЖИВАНИЕ СКАЧИВАНИЙ ==================



def list_files(folder: Path):

    folder = Path(folder)

    if not folder.is_dir():

        return {}

    res = {}

    for f in folder.iterdir():

        if f.name.endswith(".crdownload"):

            continue

        try:

            res[f.name] = f.stat().st_mtime

        except FileNotFoundError:

            pass

    return res





def wait_for_new_file(folder: Path, before: dict, timeout: int = WAIT_TIMEOUT):

    """

    Ждём новый файл в папке (таймаут 60 сек, чтобы не висеть бесконечно).

    """

    folder = Path(folder)

    t0 = time.time()

    while time.time() - t0 < timeout:

        current = list_files(folder)

        new_names = [n for n in current.keys() if n not in before]

        if new_names:

            time.sleep(1.0)

            cr = list(folder.glob("*.crdownload"))

            if not cr:

                newest = max(

                    (folder / n for n in new_names),

                    key=lambda p: p.stat().st_mtime,

                )

                return newest

        time.sleep(0.5)

    return None


def wait_for_new_file_in_folders(folders, before_map: dict, timeout: int = WAIT_TIMEOUT):
    """
    Wait for a new file in any of the provided folders.
    """
    folders = [Path(f) for f in folders if f]
    t0 = time.time()
    while time.time() - t0 < timeout:
        for folder in folders:
            current = list_files(folder)
            before = before_map.get(folder, {})
            new_names = [n for n in current.keys() if n not in before]
            if new_names:
                time.sleep(1.0)
                cr = list(folder.glob("*.crdownload"))
                if cr:
                    continue
                newest = max((folder / n for n in new_names), key=lambda p: p.stat().st_mtime)
                return newest
        time.sleep(0.5)
    return None





# ================== ОСНОВНАЯ ЛОГИКА ==================



def main():

    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)



    driver = make_driver(DOWNLOAD_ROOT, HEADLESS)

    try:

        perform_login(driver, LOGIN_EMAIL, LOGIN_PASSWORD)

        open_hse_page(driver)



        pick_exchange(driver, EXCHANGE_CODE)



        # НАИБОЛЕЕ ВАЖНЫЙ МОМЕНТ:

        # ждём, пока в списке явно появятся NASDAQ-тикеры

        wait_company_list_for_nasdaq(driver, timeout=WAIT_TIMEOUT)



        tickers = collect_tickers_from_page(driver)

        if not tickers:

            raise RuntimeError("Не удалось собрать ни одного тикера для XNAS")



        raw_count = len(tickers)



        # Мягкий фильтр: выкидываем только те, у кого тикер ОКАНЧИВАЕТСЯ на .SA

        filtered = [t for t in tickers if not t.upper().endswith(".SA")]

        if len(filtered) != raw_count:

            print(

                f"[tickers] filtered out {raw_count - len(filtered)} '.SA' tickers "

                f"(likely non-NASDAQ leftovers)."

            )

        tickers = filtered



        print(f"[tickers] using {len(tickers)} tickers for XNAS")

        save_tickers_to_txt(tickers, TICKERS_TXT_PATH)



        downloaded_with_dividends = 0

        for idx, ticker in enumerate(tickers, start=1):

            if downloaded_with_dividends >= MAX_DIVIDEND_TICKERS:
                print(f"[limit] reached {MAX_DIVIDEND_TICKERS} tickers with dividends, stopping.")
                break

            print(f"\n=== [{idx}/{len(tickers)}] {ticker} ===")



            tdir = ticker_folder(DOWNLOAD_ROOT, ticker)

            set_download_destination(driver, tdir)



            try:

                try:

                    pick_ticker(driver, ticker)

                except Exception as e:

                    print(f"   [ticker] cannot select ticker '{ticker}': {e}")

                    continue



                try:

                    reports = get_available_report_options(driver)

                except Exception as e:

                    print(f"   [reports] cannot get report list: {e}")

                    continue


                if not reports:

                    print("   [reports] no available reports for this ticker")

                    continue



                filtered_reports, missing_reports = filter_desired_reports(reports)

                if missing_reports:

                    print("   [reports] desired but unavailable:")

                    for missing in missing_reports:

                        print(f"     - {missing}")



                if not filtered_reports:

                    print("   [reports] none of the desired reports are available for this ticker")

                    continue



                dividends_norm = DIVIDENDS_REPORT_NORMALIZED



                dividends_entry = next(

                    (

                        (value, text)

                        for value, text in filtered_reports

                        if normalize_report_label(text) == dividends_norm

                    ),

                    None,

                )



                if not dividends_entry:

                    print(f"   [reports] '{DIVIDENDS_REPORT_LABEL}' not available, skipping ticker")

                    continue



                ordered_reports = [dividends_entry] + [

                    (value, text)

                    for value, text in filtered_reports

                    if normalize_report_label(text) != dividends_norm

                ]



                print(

                    f"   [reports] downloading {len(filtered_reports)} / {len(DESIRED_REPORT_NAMES)} desired reports"

                )



                dividends_timeout = False
                dividends_downloaded = False



                for value, text in ordered_reports:

                    print(f" > {text}")

                    report_norm = normalize_report_label(text)

                    try:

                        select_report(driver, value, text)



                        before_map = {tdir: list_files(tdir)}
                        if DEFAULT_DOWNLOAD_FALLBACK.exists():
                            before_map[DEFAULT_DOWNLOAD_FALLBACK] = list_files(DEFAULT_DOWNLOAD_FALLBACK)

                        click_get_data(driver)



                        new_file = wait_for_new_file_in_folders(
                            [tdir, DEFAULT_DOWNLOAD_FALLBACK], before_map, timeout=WAIT_TIMEOUT
                        )

                        if not new_file:

                            print("   no file downloaded (timeout)")

                            if report_norm == dividends_norm:

                                print("   dividends file missing after timeout, skipping ticker")

                                dividends_timeout = True

                                break

                            continue



                        stamp = dt.datetime.now().strftime("%Y%m%d")

                        ext = new_file.suffix

                        new_name = f"{base_ticker_symbol(ticker)}__{safe_name(text)}__{stamp}{ext}"

                        target = tdir / new_name



                        try:

                            new_file.replace(target)

                            print(f"   saved {target.name}")

                            if report_norm == dividends_norm:
                                dividends_downloaded = True

                        except Exception as e_move:

                            print(f"   cannot rename {new_file.name}: {e_move}")



                    except Exception as e_rep:

                        print(f"   [report-error] '{text}': {e_rep}")

                        continue



                if dividends_timeout:

                    continue

                if dividends_downloaded:
                    downloaded_with_dividends += 1


            except Exception as e_ticker:

                print(f"[ticker-fatal] {ticker}: {e_ticker}")

                continue



    finally:

        driver.quit()





if __name__ == "__main__":

    main()

