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

# Utils functions

def load_data(path, cols):
    """Load a csv file as a Pandas dataframe

    Args:
        month (str): Month of the data we wish to load

    Returns:
        pd.Dataframe: Pandas dataframe from the csv of the given month
    """
    return pd.read_csv(path, usecols=cols)

def load_data_dask(path, cols):
    """Load a csv file as a Dask dataframe

    Args:
        month (str): Month of the data we wish to load

    Returns:
        dd.Dataframe: Dask dataframe from the csv of the given month
    """
    return dd.read_csv(path, usecols=cols)


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


def plot_bar(to_plot, title, xlabel='x', ylabel='y', color='royalblue', xticks=None):
    """Given a dataframe, plots a histogram over its values

    Args:
        to_plot (pd.DataFrame): Dataframe to plot
        title (str): Title of the plot
        xlabel (str, optional): Name of the x label. Defaults to 'x'.
        ylabel (str, optional): Name of the y label. Defaults to 'y'.
        color (str, optional): Color of the plot. Defaults to 'royalblue'.
    """

    # Plot them
    _ = plt.figure()
    ax = to_plot.plot(figsize=(15, 6), kind='bar', color=color, zorder=3)
    
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

# 1.e

def view_purch_avg_time(path):
    """Compute how much time passes on average between the first view time and a purchase/addition to cart

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations

    Returns:
        float: Average value of the times that pass between the first view and a purchase/addition to cart
    """
    df = load_data(path, cols=['event_time', 'event_type', 'product_id', 'user_id'])
    df = df_parsed(df)

    df.loc[:, 'action'] = ''
    df.loc[df.event_type == 'view', 'action'] = 'view'
    df.loc[df.event_type.isin(['cart', 'purchase']),
           'action'] = 'cart_purchase'

    def view_purch_timediff(x):
        if x.shape[0] == 1:
            return None
        return max(x) - min(x)

    df_first_groups = df.groupby(['product_id', 'user_id', 'action'], sort=False).aggregate(time_first_action=pd.NamedAgg(
        column='event_time',
        aggfunc='min'
    )).reset_index()

    del df
    gc.collect()

    df_second_groups = df_first_groups.groupby(['product_id', 'user_id'], sort=False).aggregate(time_difference=pd.NamedAgg(
        column='time_first_action',
        aggfunc=view_purch_timediff
    )
    ).reset_index()

    del df_first_groups
    gc.collect

    return df_second_groups[pd.notnull(df_second_groups)['time_difference']]['time_difference'].mean()

# [RQ2] Functions

def products_for_category(path, color='darkcyan'):
    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']
    df = pd.read_csv(path, usecols=['event_type', 'category_code', 'product_id'], iterator=True, chunksize=100000)

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

def most_viewed_subcategories_month(path, num_subcat=15, plot=True, color='mediumvioletred'):
    """Plot the histogram of the viewed products for subcategory (in ascending order)

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """
    cols_to_drop = ['category', 'sub_category_2', 'sub_category_3']
    df = pd.read_csv(path, usecols=['category_code', 'event_type'], iterator=True, chunksize=100000)

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
def best_in_cat(path, cat=None):
    """Plot the histogram of the viewed products for subcategory (in ascending order)

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """
    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']
    df = pd.read_csv(path, usecols=['event_type', 'category_code', 'product_id'], iterator=True, chunksize=100000)

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

def avg_price_cat(path, category):
    """Plot the average price of the products sold by the brands in a given category

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
        category (int): Integer indicating the category for which we want the plot
    """
    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']
    df = pd.read_csv(path, usecols=['event_type', 'category_code', 'brand', 'price'], iterator=True, chunksize=100000)
    
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

def highest_price_brands(path):
    """Find, for each category, the brand with the highest average price. Return all the results in ascending order by price

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations

    Returns:
        list: List of brands sorted in ascending order by their respective price
    """
    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']

    df = pd.read_csv(path, usecols=['category_code', 'brand', 'price'], iterator=True, chunksize=100000)
    
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
    entire_df = entire_df.iloc[entire_df.groupby('category').idxmax()['price_avg']].sort_values(by='price_avg')
    gc.collect
    return entire_df


# [RQ5] functions

def avg_users(df):
    """Plot for each day of the week the hourly average of visitors the store has

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """

    week_days = []

    for _, week_day_df in df.groupby([df.event_time.dt.weekday], sort=False):
        users_num = week_day_df.groupby(
            [week_day_df.event_time.dt.hour], sort=False).count()['user_id']
        week_days.append(
            (users_num, week_day_df.event_time.iloc[0].strftime('%A')))

    plots_colors = ['royalblue', 'orange', 'mediumseagreen',
                    'crimson', 'darkcyan', 'coral', 'violet']

    # For every day of the week, plot the average number of users that visit the store each hour
    for i, (week_day, day_name) in enumerate(week_days):
        # Plot them
        plot_bar(to_plot=week_day,
                 title='Average number of users per hour - {}'.format(
                     day_name),
                 xlabel='Hour',
                 ylabel='Avg users',
                 color=plots_colors[i]
                 )
    gc.collect
    return

# [RQ6] functions

# 6.a

def conversion_rate(views, purchases):
    """Computes the conversion rate between views and purchases

    Args:
        views (pd.DataFrame): Dataframe with `event_type` view
        purchases (pd.DataFrame): Dataframe with `event_type` purchase

    Returns:
        float: Conversion rate
    """
    return purchases.groupby('product_id', sort=False).count()['event_type'].sum() / views.groupby('product_id', sort=False).count()['event_type'].sum()

def overall_conversion_rate(df, to_drop):
    """Apply conversion_rate to return the conversion rate of a dataframe

    Args:
        df (pd.DataFrame): Dataframe to use for calculations

    Returns:
        float: Conversion rate of the given dataframe
    """
    df_with_cats = subcategories_extractor(df, to_drop)
    views = views_extractor(df)
    purchases = purchases_extractor(df)
    return conversion_rate(views, purchases)

# 6.b

def category_conv_rate(df, to_drop):
    """Return and plot the conversion rate for each category of the dataframe

    Args:
        df (pd.DataFrame): Dataframe to use for calculations

    Returns:
        pd.DataFrame: Dataframe containing the conversion rates for each category
    """
    dict = {}

    df_with_cats = subcategories_extractor(df, to_drop)

    for _, frame in df_with_cats.groupby('category', sort=False):

        views = views_extractor(frame)
        purchases = purchases_extractor(frame)
        dict[frame.iloc[0]['category']] = conversion_rate(views, purchases)
        
    conv_rate_cat = pd.DataFrame.from_dict(dict.items()).rename(columns={0: 'category', 1: 'conversion rate'}).set_index('category').sort_values(by='conversion rate', ascending=False)
    plot_bar(to_plot=conv_rate_cat,
             title='Conversion rate for category',
             xlabel='category',
             ylabel='conversion rate',
             color='limegreen'
            )

    gc.collect()
    return conv_rate_cat

# [RQ7] functions

def pareto_principle(df, users_perc=20):
    """Compute the percentage of business conducted by a given percentage of the most influent users

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
        users_perc (int, optional): Percentage of users to use for the calculations. Defaults to 20.

    Returns:
        float: Percentage of business conducted by the above users
    """
    # Select all the rows that have `purchase` as event_type
    purchases = purchases_extractor(df)

    # Compute the total expenses, that are the sum of the entire column `price`
    tot_purchases = purchases['price'].sum()

    # Compute the number of unique users actually buying something (i.e., for which event_type is `purchase`)
    unique_users_number = purchases.user_id.unique().size

    # Sort in descending order the purchases for every user, using groupby and sum
    # The returning dataframe has the user that spends the most on top
    purchases_for_user = purchases.groupby(
        'user_id', sort=False).sum().sort_values('price', ascending=False)

    # Compute the number representing the (users_perc)% of the users
    # (e.g., 20% of the number of unique users if users_perc = 20)
    twnty_percent_users = int(unique_users_number / 100 * users_perc)

    # Compute the expenses made by this percentage of users that spend the most
    twenty_most = purchases_for_user.iloc[:twnty_percent_users]['price'].sum()

    # Return the percentage of expenses made by them w.r.t. to the total
    gc.collect()
    return twenty_most / (tot_purchases / 100)


def plot_pareto(df, step=10, color='darkorange'):
    """Plot the trend of the business conducted by chunks of users, with a selected step

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
        step (int, optional): Step of the percentages of users. Defaults to 10.
        color (str, optional): Plot color. Defaults to 'darkorange'.
    """
    x = np.arange(0, 105, step)
    paretos = np.array([])

    for perc in x:
        paretos = np.append(paretos, pareto_principle(df, perc))

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
