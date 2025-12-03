# BGN_LNG
LNG related pricing tools for BGN LNG desk

## Requirements so far
ideally add python and python/Scripts to Environment Variables - so that python and pip work as expected
pip install the packages:
- pandas, numpy, matplotlib, requests, xlwings
- clone virtual environment that lives in the same level as this project: "env_path"

## Steps in Excel to use the functions:

- Download python from standard sources
- Clone a python virtual environment in the same depth as this project, "{env_path}", installing required packages
- Clone this repo in the project path, "{bgn_lng_path}"
- Install xlwings: activate environment and do "xlwings install addin"
- Open Excel and save sheet in the same depth as this project, and in xlwings tab edit the fields:
    - Interpeter: Where your local python.exe interpreter lives
    - UDF Modules: So that it reads BGN_LNG.Pricers.excel_functions
    - Make sure the installed xlwings adding is in %AppData%/Microsoft/Remote/Addins
    - Enable macros in the sheet. In development tab (hidden ribbon by default) click Visual Basic -> Tools -> References -> make sure xlwings is there ticked
- Then xlwings tab -> Import Functions should work, check if bgn_kirk_price(...) exists.
