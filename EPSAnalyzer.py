from shining_pebbles import *
import os
import json
import pandas as pd
import numpy as np
import pdblp


bcon = pdblp.BCon(debug=False, port=8194, timeout=5000)
bcon.start()


def get_df_members_of_index(ticker_bbg_index='RIY Index', save=True, file_folder='dataset-bbg'):
    file_names = scan_files_including_regex(file_folder, regex=f'{ticker_bbg_index}-at{get_today("%Y%m%d")[:-2]}')
    if file_names != []:
        df = open_df_in_file_folder_by_regex(file_folder, regex=f'{ticker_bbg_index}-at{get_today("%Y%m%d")[:-2]}')
        print(f'- load complete: {ticker_bbg_index} members')
    else:
        fld = 'INDX_MEMBERS'
        members = bcon.bulkref(tickers=[ticker_bbg_index], flds=[fld])
        df = members[['ticker','field', 'value']]
        df.columns = ['ticker_bbg_index', 'flds', 'ticker']
        if save:
            save_dataset_of_subject_at(df=df, file_folder=file_folder, subject=f'{ticker_bbg_index}-{fld}', input_date=get_today("%Y%m%d"))
        get_df_names_bbg_of_index(ticker_bbg_index, save=True)
    return df

def get_tickers_bbg_index(df):
    df.columns = ['ticker_bbg_index', 'flds', 'ticker']
    tickers = list(df['ticker'])
    tickers_bbg = [f'{ticker} Equity' for ticker in tickers]
    return tickers_bbg

def get_df_names_bbg_of_index(ticker_bbg_index, save=False):
    df_members = open_df_in_file_folder_by_regex('dataset-bbg', regex=f'{ticker_bbg_index}-INDX_MEMBERS-at{get_today("%Y%m%d")[:-2]}')
    ticker_bbg_index = df_members['ticker_bbg_index'][0]
    tickers_bbg = get_tickers_bbg_index(df_members)
    flds = 'NAME'
    df = bcon.ref(tickers=tickers_bbg, flds=[flds])
    df = df[['ticker', 'value']]
    df.columns = ['ticker_bbg', flds]
    df_names = df.set_index('ticker_bbg', drop=True)
    if save:
        save_dataset_of_subject_at(df_names, file_folder='dataset-bbg', subject=f'{ticker_bbg_index}-NAME', input_date=get_today("%Y%m%d"))
        save_json_of_subject_at(df_names, file_folder='json-bbg', subject=f'{ticker_bbg_index}-NAME', input_date=get_today("%Y%m%d"))
    return df_names


def get_df_fld_from_to(tickers_bbg, fld, start_date, end_date):
    start_date= start_date.replace("-","")
    end_date= end_date.replace("-","")
    df_values = bcon.bdh(tickers=tickers_bbg, flds=[fld], start_date=start_date, end_date=end_date)
    df_values = df_values.ffill()
    
    return df_values


def get_df_fld_for_n_days(tickers_bbg, fld, n=7):
    today = get_today("%Y%m%d")
    prev_date = get_date_n_days_ago(get_today("%Y%m%d"), n=n, form="%Y%m%d")
    df_values = get_df_fld_from_to(tickers_bbg, fld, prev_date, today)
    return df_values

def get_df_fld_latest(df):
    latest_row = df.iloc[[-1]]
    dcts = latest_row.to_dict(orient='records')
    lst = []
    for dct in dcts:
        for k, v in dct.items():
            dct_new = {}        
            ticker_bbg = k[0]
            flds = k[1]
            dct_new['ticker_bbg'] = ticker_bbg
            dct_new[flds] = v
            lst.append(dct_new)
    df = pd.DataFrame(lst)    
    df = df.set_index('ticker_bbg', drop=True)
    
    return df


def get_df_cur_mkt_cap(tickers_bbg, ticker_bbg_index, n=15, save=True):
    fld = 'CUR_MKT_CAP'
    df_period = get_df_fld_for_n_days(tickers_bbg, fld, n=n)
    df_latest = get_df_fld_latest(df_period)
    df = {}
    df = {'period': df_period, 'latest': df_latest}

    if save:
        save_datasets_of_subject(df_period, df_latest, ticker_bbg_index, fld)

    return df

def get_df_best_eps(tickers_bbg, ticker_bbg_index, n=15, save=True):
    fld = 'BEST_EPS'
    df_period = get_df_fld_for_n_days(tickers_bbg, fld, n=n)
    df_latest = get_df_fld_latest(df_period)
    df = {}
    df = {'period': df_period, 'latest': df_latest}

    if save:
        save_datasets_of_subject(df_period, df_latest, ticker_bbg_index, fld)

    return df

def get_df_best_eps_next_yr(tickers_bbg, ticker_bbg_index, n=15, save=True):
    fld = 'BEST_EPS_NXT_YR'
    df_period = get_df_fld_for_n_days(tickers_bbg, fld, n=n)
    df_latest = get_df_fld_latest(df_period)
    df = {}
    df = {'period': df_period, 'latest': df_latest}

    if save:
        save_datasets_of_subject(df_period, df_latest, ticker_bbg_index, fld)

    return df

def get_df_trail_12m_eps(tickers_bbg, ticker_bbg_index, n=250, save=True):
    fld = 'TRAIL_12M_EPS'
    df_period = get_df_fld_for_n_days(tickers_bbg, fld, n=n)
    df_latest = get_df_fld_latest(df_period)
    df = {}
    df = {'period': df_period, 'latest': df_latest}
    
    if save:
        save_datasets_of_subject(df_period, df_latest, ticker_bbg_index, fld)

    return df

def preprocess_df_period(df):
    df = df.copy()
    tickers_bbg = [ticker_bbg for ticker_bbg, fld in df.columns]
    df.columns = tickers_bbg
    df.index = df.index.astype('str')
    return df

def merge_df_trail_and_df_next(df_trail, df_next):
    df_diff = df_trail.merge(df_next, how='left', left_index=True, right_index=True)
    return df_diff

def filter_df_cur_mkt_cap(df_cur_mkt_cap, lower_bound=10e3, upper_bound=None):
    df_caps_total = df_cur_mkt_cap
    df = df_caps_total[df_caps_total['CUR_MKT_CAP']>=lower_bound] 
    if upper_bound != None:
        df = df[df['CUR_MKT_CAP']<=upper_bound]
    return df

def get_tickers_bbg_of_df(df):
    tickers_bbg = list(df.index)
    return tickers_bbg

def get_df_sort_best_eps(df):
    df = preprocess_df_period(df)
    df = df.bfill()
    df = df.iloc[[-6, -1]].T
    df['rate_of_change'] = round(((df.iloc[:, 1]-df.iloc[:,0])/abs(df.iloc[:, 0]))*100,2)
    df.columns.name = None
    df.index.name = 'ticker_bbg'
    df_sorted = df.sort_values(by='rate_of_change', ascending=False)
    df_sorted['rank'] = df_sorted.apply(lambda row: df_sorted.index.get_loc(row.name) + 1, axis=1)
    return df_sorted


def get_df_sort_diff_eps(df):
    df['rate_of_change'] = round(((df.iloc[:, 1]-df.iloc[:,0])/abs(df.iloc[:, 0]))*100,2)
    df.columns.name = None
    df.index.name = 'ticker_bbg'
    df_sorted = df.sort_values(by='rate_of_change', ascending=False)
    df_sorted['rank'] = df_sorted.apply(lambda row: df_sorted.index.get_loc(row.name) + 1, axis=1)
    return df_sorted


def preprocess_df_rank(df, ticker_bbg_index):
    df_cap = open_df_in_file_folder_by_regex('dataset-eps', regex=f'{ticker_bbg_index}-CUR_MKT_CAP-at')
    df_name = open_df_in_file_folder_by_regex('dataset-bbg', regex=f'{ticker_bbg_index}-NAME-at')
    df = df.merge(df_cap, how='left', left_index=True, right_index=True)       
    df = df.merge(df_name, how='left', left_index=True, right_index=True)       
    cols = list(df.columns)
    cols_rearranged = cols[-1:] + cols[:-1]
    df = df[cols_rearranged]
    df = df.dropna()
    df_rank = df.reset_index(drop=False)
    return df_rank


def save_dataset_of_subject_at(df, file_folder, subject, input_date):
    check_folder_and_create_folder(file_folder)
    file_name = f'dataset-{subject}-at{input_date.replace("-","")}-save{get_today("%Y%m%d%H")}.csv'
    file_path = os.path.join(file_folder, file_name)
    df.to_csv(file_path, encoding='utf-8-sig')
    print(f'- save complete: {file_path}')
    return df

def save_dataset_of_subject_from_to(df, file_folder, subject, start_date=None, end_date=None):
    dates = df.index.tolist()
    start_date = format_date_to_str(dates[0], form="%Y%m%d")
    end_date = format_date_to_str(dates[-1], form="%Y%m%d")
    check_folder_and_create_folder(file_folder)
    file_name = f'dataset-{subject}-from{start_date.replace("-","")}-to{end_date.replace("-","")}-save{get_today("%Y%m%d%H")}.csv'
    file_path = os.path.join(file_folder, file_name)
    df.to_csv(file_path, encoding='utf-8-sig')
    print(f'- save complete: {file_path}')
    return df

def save_datasets_of_subject(df_period, df_latest, ticker_bbg_index, fld):
    file_folder='dataset-eps'
    dates = df_period.index.tolist()
    date_i = dates[0].strftime('%Y%m%d')
    date_f = dates[-1].strftime('%Y%m%d')
    save_dataset_of_subject_from_to(df=df_period, file_folder=file_folder, subject=f'{ticker_bbg_index}-{fld}', start_date=date_i.replace("-", ""), end_date=date_f.replace("-", ""))
    save_dataset_of_subject_at(df=df_latest, file_folder=file_folder, subject=f'{ticker_bbg_index}-{fld}', input_date=get_today("%Y%m%d"))
    return None

def save_json_of_subject_at(df, file_folder, subject, input_date):
    check_folder_and_create_folder(file_folder)
    file_name_json = f'json-{subject}-at{input_date.replace("-","")}-save{get_today("%Y%m%d%H")}.json'
    file_path_json = os.path.join(file_folder, file_name_json)
    json_names = df.reset_index().to_dict(orient='records')
    with open(file_path_json, 'w', encoding='utf-8') as file:
        json.dump(json_names, file, ensure_ascii=False, indent=4)
    print(f'- save complete: {file_path_json}')
    return None

def analyze_best_eps(ticker_bbg_index, save=True):
    check_folder_and_create_folder('dataset-analysis')
    check_folder_and_create_folder('dataset-eps')
    check_folder_and_create_folder('dataset-bbg')

    lst = scan_files_including_regex('dataset-analysis', regex=f'eps_analysis-BEST_EPS-{ticker_bbg_index}-at{get_today("%Y%m%d")}')
    if lst != []:
        df_best = open_df_in_file_folder_by_regex('dataset-analysis', regex=f'eps_analysis-{ticker_bbg_index}-BEST_EPS-at{get_today("%Y%m%d")}')
        print(f'- analysis exist: {ticker_bbg_index} best eps')
    else:
        print(f'- analysis start: {ticker_bbg_index} best eps')
        df_members = get_df_members_of_index(ticker_bbg_index, file_folder='dataset-bbg')
        tickers_bbg = get_tickers_bbg_index(df_members)

        df_cap = get_df_cur_mkt_cap(tickers_bbg, ticker_bbg_index)
        df_cap_latest = df_cap['latest']
        df_cap_filtered = filter_df_cur_mkt_cap(df_cap_latest, lower_bound=10e3)
        tickers_bbg_filtered = get_tickers_bbg_of_df(df_cap_filtered)

        df_best_eps = get_df_best_eps(tickers_bbg_filtered, ticker_bbg_index, n=15)
        df_best_period = df_best_eps['period']

        df_sort = get_df_sort_best_eps(df_best_period)
        df_best = preprocess_df_rank(df_sort, ticker_bbg_index)
    
    if save:
        save_dataset_of_subject_at(df=df_best, file_folder='dataset-analysis', subject=f'eps_analysis-{ticker_bbg_index}-BEST_EPS', input_date=get_today("%Y%m%d"))
    
    return df_best 

def analyze_diff_eps(ticker_bbg_index, save=True):
    check_folder_and_create_folder('dataset-analysis')
    check_folder_and_create_folder('dataset-eps')
    check_folder_and_create_folder('dataset-bbg')

    lst = scan_files_including_regex('dataset-analysis', regex=f'eps_analysis-TRAIL_NXT_EPS-{ticker_bbg_index}-at{get_today("%Y%m%d")}')
    if lst != []:
        df_diff = open_df_in_file_folder_by_regex('dataset-analysis', regex=f'eps_analysis-{ticker_bbg_index}-TRAIL_NXT_EPS-at{get_today("%Y%m%d")}')
        print(f'- analysis already exist: {ticker_bbg_index} best eps')
    else:
        print(f'- analysis start: {ticker_bbg_index} best eps')
        df_members = get_df_members_of_index(ticker_bbg_index, file_folder='dataset-bbg')
        tickers_bbg = get_tickers_bbg_index(df_members)

        df_cap = get_df_cur_mkt_cap(tickers_bbg, ticker_bbg_index)
        df_cap_latest = df_cap['latest']
        df_cap_filtered = filter_df_cur_mkt_cap(df_cap_latest, lower_bound=10e3)
        tickers_bbg_filtered = get_tickers_bbg_of_df(df_cap_filtered)
        df_trail_12m_eps2 = get_df_trail_12m_eps(tickers_bbg_filtered, ticker_bbg_index, n=250)
        df_trail = df_trail_12m_eps2['latest']
        df_best_eps_next_yr = get_df_best_eps_next_yr(tickers_bbg_filtered, ticker_bbg_index, n=15)
        df_next = df_best_eps_next_yr['latest']
    
        df_diff = merge_df_trail_and_df_next(df_trail, df_next)
        df_sort = get_df_sort_diff_eps(df_diff)
        df_rank = preprocess_df_rank(df_sort, ticker_bbg_index)

    if save:
        save_dataset_of_subject_at(df=df_rank, file_folder='dataset-analysis', subject=f'eps_analysis-{ticker_bbg_index}-TRAIL_NXT_EPS', input_date=get_today("%Y%m%d"))
    
    return df_rank 