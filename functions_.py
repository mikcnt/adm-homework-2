import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
import time
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import dask
import dask.dataframe as dd
from collections import defaultdict
import functools
from itertools import chain

dfs = ['./data/2019-Oct.csv',
       './data/2019-Nov.csv',
       './data/2019-Dec.csv',
       './data/2020-Jan.csv',
       './data/2020-Feb.csv',
       './data/2020-Mar.csv',
       './data/2020-Apr.csv']

# Utils functions


def iter_all_dfs(df_paths, cols):
    for i in range(len(df_paths)):
        df = pd.read_csv(df_paths[i], usecols=cols, iterator=True, chunksize=1000000)
        if i == 0:
            all_dfs = df
        else:
            all_dfs = chain(all_dfs, df)
    return all_dfs


def df_parsed(df):
    """Parse the dates as Timestamps for a dataframe

    Args:
        df (pd.DataFrame): Dataframe on which we wish to parse the dates

    Returns:
        pd.DataFrame: Dataframe with the dates parsed as Timestamps
    """
    df['event_time'] = pd.to_datetime(
        df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    return df


def purchases_extractor(df):
    """Returns a slice of the given dataframe with event_type = purchase

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: slice of the input df with just purchase instances
    """
    gc.collect
    return df.loc[df.event_type == 'purchase']

def views_extractor(df):
    """Returns a slice of the given dataframe with event_type = view

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: slice of the input df with just view instances
    """
    gc.collect
    return df.loc[df.event_type == 'view']

def cart_extractor(df):
    """Returns a slice of the given dataframe with event_type = view

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: slice of the input df with just view instances
    """
    gc.collect
    return df.loc[df.event_type == 'cart']

def rmcart_extractor(df):
    """Returns a slice of the given dataframe with event_type = view

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: slice of the input df with just view instances
    """
    gc.collect
    return df.loc[df.event_type == 'remove_from_cart']

def subcategories_extractor(df, to_drop):
    """Extracts two columns (categories and subcategories) from the column category_code

    Args:
        df (pd.DataFrame): DataFrame to use for the calculations

    Returns:
        pd.DataFrame: DataFrame with category and sub_category columns
    """
    df = df[df['category_code'].notnull()]
    df1 = df['category_code'].str.split('.', expand=True)
    df1 = df1.rename(columns={0: 'category', 1: 'sub_category_1', 2: 'sub_category_2', 3:'sub_category_3'})
    df = df.drop(columns='category_code')
    for cat in to_drop:
        if cat in df1.columns:
            df1 = df1.drop(columns=[cat])
    gc.collect()
    return pd.concat([df, df1], axis=1)


def plot_bar(to_plot, title, xlabel='x', ylabel='y', color='royalblue', xticks=None, figsize=(15, 6)):
    """Given a dataframe, plots a histogram over its values

    Args:
        to_plot (pd.DataFrame): Dataframe to plot
        title (str): Title of the plot
        xlabel (str, optional): Name of the x label. Defaults to 'x'.
        ylabel (str, optional): Name of the y label. Defaults to 'y'.
        color (str, optional): Color of the plot. Defaults to 'royalblue'.
    """

    _ = plt.figure()
    ax = to_plot.plot(figsize=figsize, kind='bar', color=color, zorder=3)
    
    if type(xticks) != type(None):
        plt.xticks(*xticks)

    # Set up grids
    plt.grid(color='lightgray', linestyle='-.', zorder=0)

    # setting label for x, y and the title
    plt.setp(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    plt.show()
    gc.collect
    return

# [RQ1] Functions

# 1.a

def plt_avg_event_session(df_paths):
    # df = pd.read_csv(path, usecols=['event_type', 'user_session'], iterator=True, chunksize=1000000)
    df = iter_all_dfs(df_paths, ['event_type', 'user_session'])
    
    num_sessions = 0
    num_views = 0
    num_cart = 0
    num_purchases = 0
    
    for frame in df:
        views = views_extractor(frame)
        cart = cart_extractor(frame)
        purchases = purchases_extractor(frame)
        
        num_views += views.groupby('user_session').event_type.count().sum()
        num_cart += cart.groupby('user_session').event_type.count().sum()
        num_purchases += purchases.groupby('user_session').event_type.count().sum()
        
        num_sessions += frame['user_session'].nunique()
        
    avg_num_operations = [num_views, num_cart, num_purchases]
    
    avg_num_operations = [num / num_sessions for num in avg_num_operations]
    
    operation_names = ['View', 'Cart', 'Purchase']
    
    avg_num_df = pd.DataFrame(avg_num_operations, columns=['average number'], index=operation_names)
    plot_bar(to_plot=avg_num_df,
             title='Average number of operations for user session',
             xlabel='Event type',
             ylabel='Average number of operations',
             color ='darkred',
            )
    gc.collect()
    return

# 1.b

def avg_view_before_cart(df_paths):
    # df = pd.read_csv(path, usecols=['event_type', 'user_id', 'product_id'], iterator=True, chunksize=1000000)
    df = iter_all_dfs(df_paths, ['event_type', 'user_id', 'product_id'])

    results = pd.DataFrame()
    for frame in df:
        frame['is_view'] = 0
        frame['is_cart'] = 0
        frame.loc[frame['event_type'] == 'view', 'is_view'] = 1
        frame.loc[frame['event_type'] == 'cart', 'is_cart'] = 1
        frame = frame.groupby(['user_id', 'product_id']).sum().reset_index()
        results = results.append(frame)
        
    results = results.groupby(['user_id', 'product_id']).sum().reset_index()
    results = results[results['is_cart'] != 0]
    avg = (results['is_view'] / results['is_cart']).mean()
    return avg

# 1.e

# def view_purch_avg_time(path):
#     """Compute how much time passes on average between the first view time and a purchase/addition to cart

#     Args:
#         df (pd.DataFrame): Dataframe to use for the calculations

#     Returns:
#         float: Average value of the times that pass between the first view and a purchase/addition to cart
#     """
#     df = load_data(path, cols=['event_time', 'event_type', 'product_id', 'user_id'])
#     df = df_parsed(df)

#     df.loc[:, 'action'] = ''
#     df.loc[df.event_type == 'view', 'action'] = 'view'
#     df.loc[df.event_type.isin(['cart', 'purchase']),
#            'action'] = 'cart_purchase'

#     def view_purch_timediff(x):
#         if x.shape[0] == 1:
#             return None
#         return max(x) - min(x)

#     df_first_groups = df.groupby(['product_id', 'user_id', 'action'], sort=False).aggregate(time_first_action=pd.NamedAgg(
#         column='event_time',
#         aggfunc='min'
#     )).reset_index()

#     del df
#     gc.collect()

#     df_second_groups = df_first_groups.groupby(['product_id', 'user_id'], sort=False).aggregate(time_difference=pd.NamedAgg(
#         column='time_first_action',
#         aggfunc=view_purch_timediff
#     )
#     ).reset_index()

#     del df_first_groups
#     gc.collect

#     return df_second_groups[pd.notnull(df_second_groups)['time_difference']]['time_difference'].mean()

# [RQ2] Functions

def products_for_category(df_paths, color='darkcyan'):
    # df = pd.read_csv(path, usecols=['event_type', 'category_code', 'product_id'], iterator=True, chunksize=100000)
    df = iter_all_dfs(df_paths, ['event_type', 'category_code', 'product_id'])
    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']


    i = 0
    for frame in df:
        frame = views_extractor(frame)
        results = subcategories_extractor(frame, cols_to_drop)
        results = results.groupby('category', sort=False).count().reset_index()
        if i == 0:
            entire_df = results
        else:
            entire_df = entire_df.append(results)
        i += 1
        del results
        gc.collect()
    entire_df =  entire_df.groupby('category').sum().sort_values(by='product_id', ascending=False)['product_id']

    # We can then plot the histogram of the number of viewed products for sub category
    plot_bar(to_plot=entire_df,
            title='Products sold for category',
            xlabel='categories',
            ylabel='products sold',
            color=color
            )
    
    gc.collect()
    return entire_df

# 2.a

def most_viewed_subcategories_month(df_paths, num_subcat=15, plot=True, color='mediumvioletred'):
    """Plot the histogram of the viewed products for subcategory (in ascending order)

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """
    # df = pd.read_csv(path, usecols=['category_code', 'event_type'], iterator=True, chunksize=100000)
    df = iter_all_dfs(df_paths, ['category_code', 'event_type'])

    cols_to_drop = ['category', 'sub_category_2', 'sub_category_3']

    i = 0
    for frame in df:
        frame = views_extractor(frame)
        results = subcategories_extractor(frame, to_drop=cols_to_drop)
        results = results.groupby('sub_category_1', sort=False).count().reset_index()

        if i == 0:
            entire_df = results
        else:
            entire_df = entire_df.append(results)
        i += 1
        del results
        gc.collect()
    entire_df =  entire_df.groupby('sub_category_1').sum().sort_values(by='event_type', ascending=False)['event_type']

    # We can then plot the histogram of the number of viewed products for sub category
    if plot:
        plot_bar(to_plot=entire_df.iloc[:num_subcat],
                title='Views for subcategory',
                xlabel='subcategories',
                ylabel='views',
                color=color
                )
    
    gc.collect()
    return entire_df

# 2.b
def best_in_cat(df_paths, cat=None):
    """Plot the histogram of the viewed products for subcategory (in ascending order)

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """

    df = iter_all_dfs(df_paths, ['event_type', 'category_code', 'product_id'])

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']
    # df = pd.read_csv(path, usecols=['event_type', 'category_code', 'product_id'], iterator=True, chunksize=100000)

    i = 0
    for frame in df:
        frame = purchases_extractor(frame)
        frame = frame[frame['category_code'].notnull()]
        if frame.empty:
            results = None
        else:
            results = subcategories_extractor(frame, to_drop=cols_to_drop)
            results = results.groupby(['category', 'product_id'], sort=False).count()
        if i == 0:
            entire_df = results
        else:
            entire_df = entire_df.append(results)
        i += 1
        del frame
        gc.collect()
    entire_df = entire_df.groupby(['category', 'product_id']).sum()
    entire_df = entire_df.groupby('category', group_keys=False, sort=False).apply(lambda x: x.sort_values(by='event_type', ascending=False).head(10)).reset_index()
    if cat == None:
        return entire_df
    
    gc.collect()
    return entire_df[entire_df['category'] == cat]


# [RQ3] Functions

# 3.a

def avg_price_cat(df_paths, category):
    """Plot the average price of the products sold by the brands in a given category

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
        category (int): Integer indicating the category for which we want the plot
    """
    df = iter_all_dfs(df_paths, ['event_type', 'category_code', 'brand', 'price'])

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']
    # df = pd.read_csv(path, usecols=['event_type', 'category_code', 'brand', 'price'], iterator=True, chunksize=100000)
    
    def f(x):
        d = {}
        d['price_sum'] = x['price'].sum()
        d['price_count'] = x['price'].count()
        return pd.Series(d, index=['price_sum', 'price_count'])
    
    i = 0
    for frame in df:
        frame = purchases_extractor(frame)
        frame = frame[frame['category_code'].notnull()]
        if frame.empty:
            results = None
        else:
            results = subcategories_extractor(frame, to_drop=cols_to_drop)
            results = results.loc[results['category'] == category].groupby('brand', sort=False).apply(f)
        if i == 0:
            entire_df = results
        else:
            entire_df = entire_df.append(results)
        i += 1
        del frame
        gc.collect()
    entire_df = entire_df.groupby(['brand']).sum()
    entire_df = (entire_df['price_sum'] / entire_df['price_count'])
    
    brands = entire_df.index
    xticks_nums = range(0, len(brands), 5)
    xticks_names = [brands[i] for i in xticks_nums]


    # Plot them
    plot_bar(to_plot=entire_df,
             title='Average price for brand',
             xlabel='brands',
             ylabel='avg price',
             xticks=(xticks_nums, xticks_names)
             )

    gc.collect
    return

# 3.b

def highest_price_brands(df_paths):
    """Find, for each category, the brand with the highest average price. Return all the results in ascending order by price

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations

    Returns:
        list: List of brands sorted in ascending order by their respective price
    """
    df = iter_all_dfs(df_paths, ['category_code', 'brand', 'price'])

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']

    # df = pd.read_csv(path, usecols=['category_code', 'brand', 'price'], iterator=True, chunksize=100000)
    
    def f(x):
        d = {}
        d['price_sum'] = x['price'].sum()
        d['price_count'] = x['price'].count()
        return pd.Series(d, index=['price_sum', 'price_count'])
    
    i = 0
    for frame in df:
        frame = frame[frame['category_code'].notnull()]
        if frame.empty:
            results = None
        else:
            results = subcategories_extractor(frame, to_drop=cols_to_drop)
        
        results = results.groupby(['category', 'brand']).apply(f).reset_index()
         
        if i == 0:
            entire_df = results
        else:
            entire_df = entire_df.append(results)
        i += 1
        del frame
        gc.collect()
        
    entire_df = entire_df.groupby(['category', 'brand']).sum()
    entire_df['price_avg'] = entire_df['price_sum'] / entire_df['price_count']
    entire_df = entire_df.drop(columns=['price_sum', 'price_count'])
    # entire_df = entire_df.groupby('category', group_keys=False, sort=False).apply(lambda x: x.sort_values(by='price_avg', ascending=False).head(1)).sort_values(by='price_avg')    
    entire_df = entire_df.iloc[entire_df.reset_index().groupby('category').idxmax()['price_avg']].sort_values(by='price_avg')
    gc.collect
    return entire_df

# [RQ4] functions

# 4.a

def monthly_profit_all_brands(df_paths, brand):
    
    entire_df = pd.DataFrame()
    for month_path in df_paths:
        
        df = pd.read_csv(month_path, usecols=['event_type', 'brand', 'price'], iterator=True, chunksize=100000)
        results = pd.DataFrame()
        for frame in df:
            frame = frame[frame['brand'].notnull()]
            if not frame.empty:
                frame = purchases_extractor(frame)
                frame = frame.groupby('brand').sum()
                results = results.append(frame)
        
        month_name = month_path.split('-')[1][:3]
        results = results.groupby('brand').sum().rename(columns={'price': month_name})
        entire_df = pd.concat([entire_df, results], axis=1)[['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']]
        
    return entire_df, entire_df[entire_df.index == brand]

# 4.b

def top_losses(all_brands, num_worst=3):
    months_diff_df = all_brands.copy()
    cols = all_brands.columns
    new_cols = []
    for i in range(0, len(cols) - 1):
        month_diff = cols[i] + '-' + cols[i + 1]
        new_cols.append(month_diff)
        months_diff_df[month_diff] = months_diff_df[cols[i]] - months_diff_df[cols[i + 1]]

    months_diff_df = months_diff_df[new_cols]
    
    max_rows = pd.DataFrame(data=[months_diff_df.max(axis=1), months_diff_df.idxmax(axis=1)]).T

    max_rows[['first_month', 'second_month']] = max_rows[1].str.split('-', expand=True)

    max_rows = max_rows.sort_values(by=0, ascending=False).drop(columns=[0, 1]).head(num_worst)
    
    for i in range(num_worst):
        brand = max_rows.index[i]
        month_1 = max_rows['first_month'][i]
        month_2 = max_rows['second_month'][i]
        value_month_1 = all_brands[all_brands.index == brand][month_1].item()
        value_month_2 = all_brands[all_brands.index == brand][month_2].item()
        percentage_lost = 100 - value_month_2 / (value_month_1 / 100)
        print('{} has lost {} between {} and {}'.format(brand, percentage_lost, month_1, month_2))

# [RQ5] functions

def avg_users(df_paths):

    df = iter_all_dfs(df_paths, ['event_time', 'user_id'])

    # df = pd.read_csv(path, usecols=['event_time', 'user_id'], iterator=True, chunksize=100000)
    
    i = 0
    n_weekdays = [0, 0, 0, 0, 0, 0, 0]
    
    def def_value():
        return pd.DataFrame()

    week_days = defaultdict(def_value)
    
    for frame in df:
        frame = df_parsed(frame)
        unique_dates = frame.event_time.dt.strftime('%d-%m-%y').unique()
        
        for date in unique_dates:
            n_weekdays[datetime.strptime(date, "%d-%m-%y").weekday()] += 1

        week_days = defaultdict(def_value)
        
        for _, week_day_df in frame.groupby(frame.event_time.dt.weekday):
            users_num = week_day_df.groupby(week_day_df.event_time.dt.hour).count()
            current_weekday = week_day_df.event_time.iloc[0].strftime('%A')
            week_days[current_weekday] = week_days[current_weekday].append(users_num['user_id']).T

    for day in week_days:
        week_days[day] = week_days[day].reset_index().groupby('index').sum()
        week_days[day] /= n_weekdays[time.strptime(day, "%A").tm_wday]

    del frame
    gc.collect()
        
        
        
    plots_colors = ['royalblue', 'orange', 'mediumseagreen',
                    'crimson', 'darkcyan', 'coral', 'violet']

    # Plot them
    for i, day in enumerate(week_days):
        plot_bar(to_plot=week_days[day],
                 title='Average number of users per hour - {}'.format(day),
                 xlabel='Hour',
                 ylabel='Avg users',
                 color=plots_colors[i]
                 )
        gc.collect
    return week_days

# [RQ6] functions

# 6.a

def purch_view(df):
    views = views_extractor(df)
    purchases = purchases_extractor(df)
    n_purchases = purchases.groupby('product_id', sort=False)['event_type'].count().sum().item()
    n_views = views.groupby('product_id', sort=False)['event_type'].count().sum().item()
    return n_purchases, n_views


def conversion_rate(df_paths):
    df = iter_all_dfs(df_paths, ['event_type', 'product_id'])

    # df = pd.read_csv(path, usecols=['event_type', 'product_id'], iterator=True, chunksize=100000)
    n_purchases = 0
    n_views = 0
    for frame in df:
        purchases, views = purch_view(frame)
        n_purchases += purchases
        n_views += views
        
    return n_purchases / n_views

# 6.b

def category_conv_rate(df_paths):
    df = iter_all_dfs(df_paths, ['event_type', 'product_id', 'category_code'])

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']
    # df = pd.read_csv(path, usecols=['event_type', 'product_id', 'category_code'], iterator=True, chunksize=1000000)
    
    def def_value():
        return np.array([0, 0], dtype=float)

    purchases_and_views = defaultdict(def_value)

    for frame in df:
        frame = frame[frame['category_code'].notnull()]
        if not frame.empty:
            frame = subcategories_extractor(frame, to_drop=cols_to_drop)
            for category_name, sub_frame in frame.groupby('category', sort=False):
                purchases_and_views[category_name] += purch_view(sub_frame)
    
    cat_df = pd.DataFrame(purchases_and_views).T.rename(columns={0: 'purch_num', 1: 'views_num'})
    cat_df['conversion_rate'] = cat_df['purch_num'] / cat_df['views_num']
    cat_df = cat_df.drop(columns=['purch_num', 'views_num'])
    
    plot_bar(to_plot=cat_df,
             title='Conversion rate for category',
             xlabel='category',
             ylabel='conversion rate',
             color='limegreen'
            )

    gc.collect()
    return cat_df

# [RQ7] functions

def pareto_principle(df_paths, users_perc=20):
    df = iter_all_dfs(df_paths, ['event_type', 'price', 'user_id'])
    
    # df = pd.read_csv(path, usecols=['event_type', 'price', 'user_id'], iterator=True, chunksize=1000000)
    
    initial_results = {
            'tot_purchases': 0,
            'purchases_for_user': pd.DataFrame(),
            'unique_users': np.array([])
        }

    def accumulate_data(prev, frame):
        purchases = purchases_extractor(frame)
        return {
            'tot_purchases': prev['tot_purchases'] + purchases['price'].sum(),
            'purchases_for_user': prev['purchases_for_user'].append(purchases.groupby('user_id', sort=False).sum()),
            'unique_users': np.append(prev['unique_users'], purchases['user_id'].unique())
        }

    tot_purchases, purchases_for_user, unique_users = functools.reduce(accumulate_data, df, initial_results).values()

    unique_users_number = np.unique(unique_users).size
    
    # 2) Results is now composed by the chunks of dataframes on which we've done the operations
    # but, merging, we've created new rows with the same user_id. This means that we have to
    # groupby again and sum over them. After that, just sort the values in descending order
    purchases_for_user = purchases_for_user.groupby('user_id', sort=False).sum().sort_values('price', ascending=False)
    
    
    # Compute the number representing the (users_perc)% of the users
    # (e.g., 20% of the number of unique users if users_perc = 20)
    twnty_percent_users = int(unique_users_number / 100 * users_perc)
    
    # Compute the expenses made by this percentage of users that spend the most
    twenty_most = purchases_for_user.iloc[:twnty_percent_users]['price'].sum()
    
    # Return the percentage of expenses made by them w.r.t. to the total
    gc.collect()
    return twenty_most / (tot_purchases / 100)


def plot_pareto(df_paths, step=20, color='darkorange'):
    """Plot the trend of the business conducted by chunks of users, with a selected step

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
        step (int, optional): Step of the percentages of users. Defaults to 10.
        color (str, optional): Plot color. Defaults to 'darkorange'.
    """
    x = np.append(np.arange(5, 100, step), 100)
    paretos = np.array([])

    for perc in x:
        paretos = np.append(paretos, pareto_principle(df_paths, perc))

    paretos_df = pd.DataFrame(index=x, data=paretos).rename(
        columns={0: 'Pareto Behaviour'})

    plot_bar(to_plot=paretos_df,
             title='Pareto principle w.r.t. percentage of users - step of {}'.format(
                 step),
             xlabel='Percentage of users considered',
             ylabel='Percentage of business conducted by users',
             color=color)

    gc.collect
    return
