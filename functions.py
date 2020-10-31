import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
import time
import matplotlib.pyplot as plt
from datetime import datetime
import gc

# Utils functions

def load_data(month):
    """Load a csv file as a Pandas dataframe

    Args:
        month (str): Month of the data we wish to load

    Returns:
        pd.Dataframe: Pandas dataframe from the csv of the given month
    """
    return pd.read_csv("./data/2019-{}.csv".format(month))


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


def plot_bar(to_plot, title, xlabel='x', ylabel='y', color='royalblue'):
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

    # Set up grids
    plt.grid(color='lightgray', linestyle='-.', zorder=0)

    # setting label for x, y and the title
    plt.setp(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    plt.show()
    gc.collect
    return

# [RQ1] Functions

# 1.e

def view_purch_avg_time(df):
    """Compute how much time passes on average between the first view time and a purchase/addition to cart

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations

    Returns:
        float: Average value of the times that pass between the first view and a purchase/addition to cart
    """

    df.loc[:, 'action'] = ''
    df.loc[df.event_type == 'view', 'action'] = 'view'
    df.loc[df.event_type.isin(['cart', 'purchase']),
           'action'] = 'cart_purchase'

    def view_purch_timediff(x):
        if x.shape[0] == 1:
            return None
        return max(x) - min(x)

    df_first_groups = df.groupby(['product_id', 'user_id', 'action']).aggregate(time_first_action=pd.NamedAgg(
        column='event_time',
        aggfunc='min'
    )).reset_index()

    df_second_groups = df_first_groups.groupby(['product_id', 'user_id']).aggregate(time_difference=pd.NamedAgg(
        column='time_first_action',
        aggfunc=view_purch_timediff
    )
    ).reset_index()

    gc.collect
    return df_second_groups[pd.notnull(df_second_groups)['time_difference']]['time_difference'].mean()

# [RQ3] Functions

# 3.a

def avg_price_cat(df, category):
    """Plot the average price of the products sold by the brands in a given category

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
        category (int): Integer indicating the category for which we want the plot
    """

    # Create a dataframe with just the purchases as event_type
    df_purchases = purchases_extractor(df)

    # Compute the average prices
    avg_prices = df_purchases.loc[df_purchases['category_id'] == category].groupby(
        'brand').mean()['price']

    # Plot them
    plot_bar(to_plot=avg_prices,
             title='Average price for brand',
             xlabel='brands',
             ylabel='avg price'
             )

    gc.collect
    return

# 3.b

def highest_price_brands(df):
    """Find, for each category, the brand with the highest average price. Return all the results in ascending order by price

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations

    Returns:
        list: List of brands sorted in ascending order by their respective price
    """
    # Create a dataframe with just the purchases as event_type
    df_purchases = purchases_extractor(df)

    # Select only the ones where the `brand` column is not empty
    df_notnull_purchases = df_purchases.loc[df_purchases.brand.notnull()]

    # Instantiate a dictionary
    high_brands = {}

    # Fill with the category number as key and its (brand, price) as values
    # brand selected will be the one with avg highest price in the selected category
    # Iterate on the dframes created by the groupby on the category
    for _, category_frame in df_notnull_purchases.groupby('category_id'):
        category_num = category_frame['category_id'].iloc[0]

        # For each frame, group on the brand and transform with mean
        avg_prices_cat = category_frame.groupby('brand').mean().reset_index()

        # Select row index with highest price
        idx = avg_prices_cat['price'].argmax()

        # Extract brand name and respective price for each category
        category_brand = avg_prices_cat.iloc[idx]['brand']
        category_price = avg_prices_cat.iloc[idx]['price']

        # Fill the dictionary
        high_brands[category_num] = (category_brand, category_price)

    dict_values = list(high_brands.values())

    # Sort the dictionary values (brand, price) w.r.t. the price
    dict_values.sort(key=lambda x: x[1])

    # Return the keys, which are the actual brands sorted
    gc.collect
    return [x[0] for x in dict_values]

# [RQ5] functions

def avg_users(df):
    """Plot for each day of the week the hourly average of visitors the store has

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """

    week_days = []

    for _, week_day_df in df.groupby([df.event_time.dt.weekday]):
        users_num = week_day_df.groupby(
            [week_day_df.event_time.dt.hour]).count()['user_id']
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
