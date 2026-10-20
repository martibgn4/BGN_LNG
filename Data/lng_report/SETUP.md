# Running the daily settlement mail on a new machine

The settlement snapshot (`_run_settlements`) pulls TTF, Brent, Henry Hub and the
Baltic freight routes from Bloomberg, the NWE discount and Spark freight curves
from the Spark API, and JKM from a settlement master built off screenshots. It
then mails the table to `lng@bgn-int.com` through the local Outlook client.

Anyone with a logged-in Bloomberg Terminal can run it. The Spark credentials
ship with the repo, so there is nothing to request for that.

## 1. Install

```bash
git clone https://github.com/martibgn4/BGN_LNG.git
cd BGN_LNG
pip install -r requirements.txt
```

`blpapi` is **not on PyPI** and the line above will not install it. Get it from
Bloomberg's own index:

```bash
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

Needs Python 3.9+, Windows, and a Bloomberg Terminal running and logged in on
the same machine. Delivery goes through the Outlook desktop client over COM, so
Outlook must be installed and signed in too.

## 2. Check the machine before you trust it

```bash
python -m Data.lng_report.preflight
```

This checks every prerequisite in one pass — packages, `blpapi`, the terminal,
Spark auth, the JKM master, Outlook, and the output folder — and prints the fix
next to anything missing. Exit code is 0 when the mail can be sent.

`[FAIL]` means no mail at all. `[WARN]` means it still sends, with something
degraded (a blank column, or the published copy skipped).

## 3. Build it without sending

```bash
python -m Data.lng_report.daily_market_data_reporting --dry-run
```

Builds the table, the HTML and the workbook, prints the table, and sends
nothing. Always do this first on a new machine.

## 4. Send it

```bash
python -m Data.lng_report.daily_job --only settlements
```

Goes to `lng@bgn-int.com; vasileios.giannoutsos@bgn-int.com`. To send somewhere
else while testing:

```bash
python -m Data.lng_report.daily_market_data_reporting --to "you@bgn-int.com"
```

## What is machine-specific, and how it resolves

Nothing below needs setting on a normal BGN machine. Each falls back on its own.

| What | Resolution order |
|---|---|
| Spark credentials | `SPARK_CREDENTIALS_PATH` → `adhoc_scripts/client_credentials.local.csv` (gitignored, for your own Spark client) → `adhoc_scripts/client_credentials.csv` (ships with the repo) → `SPARK_CLIENT_ID`/`SPARK_CLIENT_SECRET` |
| Published copy | `BGN_LNG_REPORT_DIR` → the desk's OneDrive folder, when it is actually synced → `<repo>/output` |
| JKM master | `JKM_MASTER_PATH` → `Data/jkm_settlements/master.csv` → the desk's own path |

### The JKM column will be blank for you

The JKM settlement master is Platts-derived price data, so it is not in the
repo. Without it the JKM column comes back empty and the mail header says
`JKM: unavailable` in red — every other column is unaffected. To fill it, get a
copy of `master.csv` from the desk and either drop it at
`Data/jkm_settlements/master.csv` or point `JKM_MASTER_PATH` at it.

### Columns that are legitimately short

Not every blank is a failure, and the mail says so in its footer:

- **Spark** (NWE discount, Spark30, Spark25) — the curves only run about 16
  months out. Everything beyond that is blank because nothing is quoted.
- **Baltic** (BLNG1/2/3) — quoted to Dec-29 only.
- **Brent** — the contract for delivery month M stops trading at the end of
  M-2, so the nearest row is empty once that month has rolled.

## Troubleshooting

**`ModuleNotFoundError: No module named 'blpapi'`** — step 1's second command
was skipped.

**`ModuleNotFoundError: No module named 'Data'`** — run from the repo root, not
from inside `Data/lng_report/`.

**Bloomberg request fails or returns nothing** — the terminal is not logged in,
or this machine is not entitled to futures data. `preflight` distinguishes the
two.

**Spark rejects the credentials** — they have been rotated since the copy in
the repo was committed. Ask the desk for a current client id and secret, put
them in `adhoc_scripts/client_credentials.local.csv`, and that copy wins over
the committed one.

**Nothing is published but the mail arrives** — expected off the BGN network:
the shared drive is not mounted, so the copy went to `<repo>/output`. The table
is in the mail body regardless.
