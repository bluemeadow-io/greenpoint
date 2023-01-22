import pandas as pd
import numpy as np
import time
import warnings  as wrn


MONTH_CODES = "FGHJKMNQUVXZ"

MONTH_NAMES = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

MONTH_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

MONTH_NAME_TO_CODE = {k: v for k, v in zip(MONTH_NAMES, MONTH_CODES)}

FIELDS_MAP = {
    "Trade Date": "date",
    "Risk Free Interest Rate": "RATE",
    "Open Implied Volatility": "PRICE_OPEN",
    "Last Implied Volatility": "PRICE_LAST",
    "High Implied Volatility": "PRICE_HIGH",
    "Previous Close Price": "PRICE_CLOSE_PREV",
    "Close Implied Volatility": "IMPLIEDVOL_BLACK",
    "Strike Price": "STRIKE",
    "Option Premium": "PREMIUM",
    "General Value6": "UNDL_PRICE_SETTLE",
    "General Value7": "UNDL_PRICE_LAST",
}

FLOAT_FIELDS = [
    "PRICE_OPEN",
    "PRICE_LAST",
    "PRICE_HIGH",
    "PRICE_CLOSE_PREV",
    "IMPLIEDVOL_BLACK",
    "PREMIUM",
    "RATE",
    "STRIKE",
    "UNDL_PRICE_SETTLE",
    "UNDL_PRICE_LAST",
]


def transform(raw_data_: pd.DataFrame, instruments_: pd.DataFrame) -> pd.DataFrame:

    """
    Create a function called transform that returns a normalized table.
    Do not mutate the input.
    The runtime of the transform function should be below 1 second.

    :param raw_data_: dataframe of all features associated with instruments, with associated timestamps
    :param instruments_: dataframe of all traded instruments
    """

    """
    Create a column called contract, which is a copy of the Term column. 
    If there are nulls in the Term column, fill it with the Period column.
    """
    raw_data_['Contract'] = raw_data_['Term']
    null_idx = pd.isnull(raw_data_['Term'])
    raw_data_.loc[null_idx, 'Contract'] = raw_data_['Period'].loc[null_idx]

    """
    Check for nulls in Trade Date column. 
    Raise a warning if any nulls exist and drop the rows with
    null Trade Date.

    Check for expired instruments. 
    An instrument is expired if the 
    Expiration Date is older than the Trade Date. 
    Raise a warning if there are expired instruments 
    and drop the rows with expired instruments.

    Check for nulls in contract column. Raise a warning 
    if nulls exist and drop the rows with null contract name.
    """
    raw_data_['Trade Date'] = pd.to_datetime(raw_data_['Trade Date'])
    raw_data_['Expiration Date'] = pd.to_datetime(raw_data_['Expiration Date'])
    drop_conditions = {
        'Null Trade Date': pd.isnull(raw_data_['Trade Date']),
        'Expired Instruments': raw_data_['Trade Date'] > raw_data_['Expiration Date'],
        'Null Contract': pd.isnull(raw_data_['Contract']),
    }
    for name, cond in drop_conditions.items():    
        if any(cond):
            wrn.warn(f"{name} are present! Dropping!")
            raw_data_ = raw_data_[~cond]

    """
    Parse the RIC base and moneyness from the RIC column 
    and merge instruments dataframe on the base.
    Example: 1BO50Nc1=R -> moneyness = 50, base=1BO
    Example: 1BO100Nc1O=R -> moneyness = 100, base=1BO
    """
    def parse_ric(ric):
        base = None
        moneyness = None
        for i, c in enumerate(ric[2:]):
            if c.isdigit():
                base = ric[:2+i]
                moneyness = ric[2+i:].split('Nc')[0]
                break
        return (base, moneyness)
    raw_data_['base'] = raw_data_['RIC'].map(lambda x: parse_ric(x)[0])
    raw_data_['moneyness'] = raw_data_['RIC'].map(lambda x: parse_ric(x)[1])
    raw_data_ = raw_data_.set_index('base').join(
        instruments_.set_index('Base'), 
        lsuffix=' _Drop R', 
        rsuffix=' _Drop B').reset_index()
    for c in raw_data_.columns:
        if '_Drop' in c:
            del raw_data_[c]
    
    """
    Create two columns called contract_year and contract_month. 
    For contract_year, parse the year from Expiration Date. 
    For contract_month, parse the month from Period (JAN, FEB, etc.). 

    If the last digit of the contract_year does not equal the 
    last digit of the Period column, increment contract_year by 1. 
    You may or may not see Jan/Feb contracts that do not match the 
    year of the expiration date. For instance, a Jan 2021 contract 
    may expire in December 19, 2020.
    """
    exp_date = pd.to_datetime(raw_data_['Expiration Date'])
    raw_data_['contract_year'] = exp_date.map(lambda x: x.year)
    raw_data_['contract_month'] = raw_data_['Period'].map(
                        lambda x: MONTH_NAMES.index(x[:3])+1)
    roll_year = raw_data_.apply(
        lambda x: 1 if str(x['contract_year'])[-1] != x['Period'][-1] else 0, axis=1)
    raw_data_['contract_year'] = raw_data_['contract_year'] + roll_year

    """
    Create a column called month_code that maps 
    contract_month using the MONTH_NAME_TO_CODE dict.
    """
    raw_data_['month_code'] = \
            raw_data_['contract_month'].map(\
                lambda x: MONTH_NAME_TO_CODE[MONTH_NAMES[x-1]])

    """
    Rename some columns using FIELD_MAP
    """
    raw_data_.rename(columns = FIELDS_MAP, inplace = True)

    """
    Create a column called symbol that is a concatenation of 
    "FUTURE_VOL_", Exchange, Bloomberg Ticker, 
    month_code, contract_year, and moneyness. 
    Single character Bloomberg Tickers are special.
    Example: FUTURE_VOL_CBT_BOM2022_50, FUTURE_VOL_CBT_W_N2023_150
    """
    def get_symbol(x):
        myyyy = x['month_code'] + str(x['contract_year'])
        if len(x['Bloomberg Ticker']) == 1:
            return f"FUTURE_VOL_{x['Exchange']}_{x['Bloomberg Ticker']}_{myyyy}_{x['moneyness']}"
        else:
            return f"FUTURE_VOL_{x['Exchange']}_{x['Bloomberg Ticker']}{myyyy}_{x['moneyness']}"
    raw_data_['symbol'] = raw_data_.apply(get_symbol, axis=1)

    """
    Make sure all float fields are actuall float field, convert if necessary
    """
    for fn in FLOAT_FIELDS:
        if raw_data_[fn].dtype != np.dtype('float64'):
            print('fixing type for:', fn, raw_data_[fn].dtype)
            raw_data_[fn] = pd.to_numeric(raw_data_[fn].map(lambda x: x.replace(',',''))).astype(float)

    output = pd.melt(raw_data_, id_vars=['date','symbol'], value_vars=FLOAT_FIELDS)
    output['source'] = 'refinitiv'
    output.rename(columns = {'variable':'field'}, inplace = True)
    output = output[['date', 'symbol', 'source', 'field', 'value']]
    return output

if __name__ == '__main__':
    raw_data = pd.read_csv("raw_data.csv")
    instruments = pd.read_csv("instruments.csv")
    st = time.process_time()
    output = transform(raw_data, instruments)
    et = time.process_time()
    print(f"Wall time: {100 * (et-st)} ms")
    expected_output = pd.read_csv(
        "expected_output.csv",
        index_col=0,
        parse_dates=['date']
    )
    pd.testing.assert_frame_equal(output, expected_output)