import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import squarify
import time
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import gc
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


def iter_all_dfs(df_paths, cols, chunksize=1000000):
    """Create a Pandas iterable given a list of paths.

    Args:
        df_paths (str): List containing the paths of the csv files.
        cols (list): List of columns to include in the dataframes of the iterable.
        chunksize (int, optional): Chunksize of the iterable. Defaults to 100000.

    Returns:
        pd.io.parsers.TextFileReader: Pandas dataframe iterable to work with a list of csv files in chunks.
    """
    for i in range(len(df_paths)):
        df = pd.read_csv(df_paths[i], usecols=cols, iterator=True, chunksize=chunksize)
        if i == 0:
            all_dfs = df
        else:
            all_dfs = chain(all_dfs, df)
    return all_dfs


def df_parsed(df):
    """Parse the dates of a dataframe as Timestamps for a dataframe.

    Args:
        df (pd.DataFrame): Dataframe on which we wish to parse the dates.

    Returns:
        pd.DataFrame: Dataframe with the event_time column parsed as Timestamps.
    """
    df['event_time'] = pd.to_datetime(
        df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    return df


def only_purchases(df):
    """Select only the rows of the dataframes with purchase in the `event_type` column.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Slice of the input df with just purchase instances.
    """
    gc.collect
    return df.loc[df.event_type == 'purchase']

def only_views(df):
    """Select only the rows of the dataframes with view in the `event_type` column.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Slice of the input df with just view instances.
    """
    gc.collect
    return df.loc[df.event_type == 'view']

def only_carts(df):
    """Select only the rows of the dataframes with cart in the `event_type` column.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Slice of the input df with just cart instances.
    """
    gc.collect
    return df.loc[df.event_type == 'cart']


def subcategories_extractor(df, to_drop):
    """Split the `category_code` column in multiple columns: `category`, `sub_category_1`, `sub_category_2`, `sub_category_3`.
    Then, drop the to_drop columns. Drop the rows of the dataframe where the `category_code` column is missing.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with the category and the subcategories columns, without the to_drop ones.
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
    """Plot a histogram over a dataframe.

    Args:
        to_plot (pd.DataFrame): Dataframe to plot.
        title (str): Title of the plot.
        xlabel (str, optional): Name of the x label. Defaults to 'x'.
        ylabel (str, optional): Name of the y label. Defaults to 'y'.
        color (str, optional): Color of the plot. Defaults to 'royalblue'.
        xticks (list of lists, optional): Ticks of the plot. Defaults to None, which means that all the ticks are plotted.
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
    """Plot the average number of times users perform each of the following operations: `view`, `cart`, `purchase`.
    Skip the `remove_from_cart`, since there isn't any.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to plot the statistics.
    """
    # Load data in an iterable to work in chunks
    df = iter_all_dfs(df_paths, ['event_type', 'user_session'])
    
    # Initialize values
    num_sessions = 0
    num_views = 0
    num_cart = 0
    num_purchases = 0
    
    # Since we loaded the dataframes as an iterable, we need to iterate in it and perform
    # the operations on each of the chunks.
    for frame in df:
        # Extract the views, carts and purchases
        views = only_views(frame)
        cart = only_carts(frame)
        purchases = only_purchases(frame)
        
        # Number of views, carts and purchases for each chunk is obtained with a single groubpy
        num_views += views.groupby('user_session').event_type.count().sum()
        num_cart += cart.groupby('user_session').event_type.count().sum()
        num_purchases += purchases.groupby('user_session').event_type.count().sum()
        
        # We also need to keep track of the number of sessions
        num_sessions += frame['user_session'].nunique()
    

    avg_num_operations = [num_views, num_cart, num_purchases]
    
    # To extract the mean average of each operation, we need to divide for the number of unique sessions
    avg_num_operations = [num / num_sessions for num in avg_num_operations]
    
    operation_names = ['View', 'Cart', 'Purchase']
    
    # Since our plot function receives a Pandas df as input, create one using the statistics we just extracted
    avg_num_df = pd.DataFrame(avg_num_operations, columns=['average number'], index=operation_names)

    # Plot
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
    """Compute the number of times a user views a product before adding it to the cart.
    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.
    Returns:
        float: Average number of the times a user views a product before adding it into the cart.
    """

    # Initialize an empty dictionary on which we are going to put
    # the statistics for each month.
    averages = {}
    for month_path in df_paths:
        # Read the month-dataframe
        df = pd.read_csv(month_path, usecols=['event_type', 'user_id', 'product_id'], iterator=True, chunksize=1000000)
        
        # Initialize an empty dataframe on which we are going to append
        # the statistics for each chunk.
        results = pd.DataFrame()
        
        # Get the month name using simple parsing techniques
        month_name = month_path.split('-')[1][:3]
        
        # Since we loaded the dataframe as an iterable, we need to iterate in it and perform
        # the operations on each of the chunks.
        for frame in df:
            # Create two new columns, one for the view instances and one for the cart instances
            frame['is_view'] = 0
            frame['is_cart'] = 0

            # To keep track of what's what, set to 1 the respective new column where we find
            # one of these instances
            frame.loc[frame['event_type'] == 'view', 'is_view'] = 1
            frame.loc[frame['event_type'] == 'cart', 'is_cart'] = 1

            # Sum over the user and the product so that we can analyze the new columns
            # and find out how many times the user viewed/carted a certain product
            frame = frame.groupby(['user_id', 'product_id']).sum().reset_index()

            # Since we're working in chunks, we need to append our results to a new dataframe
            # each time. This is what results is used for
            results = results.append(frame)
            results = results.groupby(['user_id', 'product_id']).sum().reset_index()
            
            del frame
            gc.collect()

        # Once we've finished, we need to use the groupby again, since we could have created
        # new rows with the same user and product over and over (we're working with chunks!)
        results = results.groupby(['user_id', 'product_id']).sum().reset_index()

        # We want only the elements for which the product has actually put into the cart
        results = results[results['is_cart'] != 0]

        # Extract the average by just using the mean
        avg = (results['is_view'] / results['is_cart']).mean()
        
        averages[month_name] = avg

    return averages

def averages_printer(averages_dict):
    """Prints the average for each month.

    Args:
        averages_dict (dict): Dictionary containing the averages for each month.
    """
    for key, value in averages_dict.items():
        print("During {}, the average number of views for cart is {}.".format(key, value))
    return

# 1.c

def purchase_after_cart_prob(df_paths):
    """Compute the probability that products added once to the cart are bought.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.

    Returns:
        float: Probability that products are bought after being added to the cart once.
    """
    # Load data in an iterable to work in chunks
    df = iter_all_dfs(df_paths, ['event_type', 'user_id', 'product_id'])

    # Initialize an empty dataframe on which we are going to append
    # the statistics for each chunk.
    results = pd.DataFrame()
    
    # Since we loaded the dataframes as an iterable, we need to iterate in it and perform
    # the operations on each of the chunks.
    for frame in df:
        # We don't need the views instances
        frame = frame[frame['event_type'] != 'view']

        # Create two new columns, ore for the cart and one for the purchase instances
        frame['is_cart'] = 0
        frame['is_purchase'] = 0

        # To keep track of what's what, set to 1 the respective new column where we find
        # one of these instances
        frame.loc[frame['event_type'] == 'cart', 'is_cart'] = 1
        frame.loc[frame['event_type'] == 'purchase', 'is_purchase'] = 1

        # Sum over the user and the product so that we can analyze the new columns
        # and find out how many times the user carted/purchased a certain product
        frame = frame.groupby(['user_id', 'product_id']).sum().reset_index()

        # Since we're working in chunks, we need to append our results to a new dataframe
        # each time. This is what results is used for
        results = results.append(frame)
    
    # Once we've finished, we need to use the groupby again, since we could have created
    # new rows with the same user and product over and over (we're working with chunks!)
    results = results.groupby(['user_id', 'product_id']).sum()

    # We just want the items that are put in the cart once
    results = results[results['is_cart'] == 1]

    # To compute the probability, we just need to divide the number of times an item is put
    # into cart once and bought, for the total number of times it is put into cart
    cart_purch_num_1 = results[(results['is_purchase'] == 1) & (results['is_cart'] == 1)].shape[0]
    cart_total_num_1 = results.shape[0]

    return cart_purch_num_1 / cart_total_num_1

# 1.d

def check_event_types(df_paths):
    """Compute the list of unique elements in the `event_type` column for each dataframe
    in the list of paths.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.
    """
    # Iterate through the path of the dataframes
    for path in df_paths:
        print('Unique values in the `event_type` column for dataframe `{}` are:'.format(path.split('/')[-1]))
        df = pd.read_csv(path, usecols=['event_type'], iterator=True)
        possible_event_types = np.array([])
        for frame in df:
            possible_event_types = np.append(possible_event_types, frame.event_type.unique())
            possible_event_types = np.unique(possible_event_types)
        
        print(possible_event_types)
        print('---'*10)

# 1.e

def avg_time_view_action(df_paths):
    """Compute average time between the first view and purchase/cart.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.
    """
    # Load data in an iterable to work in chunks
    df = iter_all_dfs(df_paths, ['event_time', 'event_type', 'user_session'], chunksize=1000000)
    
    # Initialize three dataframes, one for the views statistics, one for the carts and one for the purchases
    views_results = pd.DataFrame()
    carts_results = pd.DataFrame()
    purchases_results = pd.DataFrame()
    # Working with chunks
    for frame in df:
        # For each chunk we need to parse the dates
        frame = df_parsed(frame)
        
        # Create three new dataframes, each one containing only one type of event (view, cart, purch)
        # For each of these ones, we drop the duplicates, so that we just see the first action in each session
        views = only_views(frame).drop_duplicates('user_session')
        carts = only_carts(frame).drop_duplicates('user_session')
        purchases = only_purchases(frame).drop_duplicates('user_session')
        
        # Since we're working with chunks, we need to append each time and create a big dataframe for the stats
        views_results = views_results.append(views)
        carts_results = carts_results.append(carts)
        purchases_results = purchases_results.append(purchases)
        
        # Free some memory since this function is very low-memory efficient
        del frame, views, carts, purchases
        gc.collect()
    # Since we appended multiple times the dataframes, even if the initial dataframe was
    # sorted by the event time, this new one is not necessarily.
    # After the sorting, drop again the duplicates for the same reason: working with chunks
    views_results = views_results.sort_values(by='event_time').drop_duplicates('user_session')
    carts_results = carts_results.sort_values(by='event_time').drop_duplicates('user_session')
    purchases_results = purchases_results.drop_duplicates('user_session')
    
    # Now we can do an inner join with the merge function, obtaining a big dataframe
    # with the event_time of the view, and event time for the cart (views_carts)
    # and one with event_time of view and cart (views_purchases)
    # Remember to join on the user session!
    views_carts = pd.merge(views_results, carts_results, on='user_session')
    views_purchases = pd.merge(views_results, purchases_results, on='user_session')
    
    # We want to compute the average time between these actions, so we need their difference
    time_diff_carts = views_carts['event_time_y'] - views_carts['event_time_x']
    time_diff_purchases = views_purchases['event_time_y'] - views_purchases['event_time_x']

    # Just compute the mean distance and then format so it is readable
    avg_carts = str(timedelta(seconds=time_diff_carts.mean().total_seconds())).split('.')[0]
    avg_purchases = str(timedelta(seconds=time_diff_purchases.mean().total_seconds())).split('.')[0]
    
    # Print the results
    print('Average time between first view and first cart is {}.'.format(avg_carts))
    print('Average time between first view and first purchase is {}.'.format(avg_purchases))
    
    return

# [RQ2] Functions

def products_for_category(df_paths):
    """Plot a histogram representing the number of sold products per category.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to plot the statistics.

    Returns:
        pd.DataFrame: Dataframe containing the plotted statistics.
    """
    # Load the dataframe as an iterable to use chunks
    df = iter_all_dfs(df_paths, ['event_type', 'category_code', 'product_id'])

    # At some point, we're going to extract the category and subcategories.
    # We just need the category, so drop the subcategories.
    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']

    results = pd.DataFrame()
    # Iterate on the chunks
    for frame in df:
        # Extract only the purchases 
        frame = only_purchases(frame)

        # Extract the the category from the `category_code` column
        frame = subcategories_extractor(frame, cols_to_drop)

        if not frame.empty:
        # Count the number of sold products for category
            frame = frame.groupby('category', sort=False).count().reset_index()

            # Append to a new df for each chunk
            results = results.append(frame)

        del frame
        gc.collect()

    # Once we've finished working with the chunks, we need to sum over the category again
    # since appending could've created new rows with the same category
    # Once we've done that, we just sort the number of sold products for a better looking histogram
    results =  results.groupby('category').sum().sort_values(by='product_id', ascending=False)['product_id']

    # Plot
    plot_bar(to_plot=results,
            title='Products sold for category',
            xlabel='categories',
            ylabel='products sold',
            color='darkcyan'
            )
    
    # Return same results
    return results

# 2.a

def most_viewed_subcategories_month(df_paths, num_subcat=15):
    """Plot the most viewed subcategories.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to plot the statistics.
        num_subcat (int, optional): Number of subcategories to include in the plot. Defaults to 15.

    Returns:
        pd.DataFrame: Dataframe with the plotted statistics.
    """
    # Iterable of the dataframes, to use chunks
    df = iter_all_dfs(df_paths, ['category_code', 'event_type'])

    # List of the columns we don't want when extracting categories and subcategories
    # In this case we just want to work with the first subcategory, drop the others
    cols_to_drop = ['category', 'sub_category_2', 'sub_category_3']

    # Initialize the resulting dataframe after the chunks
    results = pd.DataFrame()

    # Iterate through the chunks of the df
    for frame in df:
        # Extract only the views
        frame = only_views(frame)

        # Extract only the first subcategory
        frame = subcategories_extractor(frame, to_drop=cols_to_drop)
        if not frame.empty:
            # Compute statistics with a groupby and count oer the first sub_category
            frame = frame.groupby('sub_category_1', sort=False).count().reset_index()

            # Append to the resulting dataframe
            results = results.append(frame)

        del frame
        gc.collect()
    
    # Compute again the sum over the sub_category since we could've added new rows
    # with the same sub_category
    results =  results.groupby('sub_category_1').sum().sort_values(by='event_type', ascending=False)['event_type']

    # Plot
    plot_bar(to_plot=results.iloc[:num_subcat],
            title='Views for subcategory',
            xlabel='subcategories',
            ylabel='views',
            color='mediumvioletred'
            )
    
    # Return the same plotted statistics
    return results

# 2.b
def best_in_cat(df_paths, cat=None):
    """Compute the 10 most sold products per category.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.
        cat (string, optional): If given, show the 10 most sold products of the category. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of the 10 most sold products per category, or for a single category if specified. 
    """
    # Load data in an iterable to work in chunks
    df = iter_all_dfs(df_paths, ['event_type', 'category_code', 'product_id'])

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']

    results = pd.DataFrame()

    for frame in df:
        frame = only_purchases(frame)

        # Drop the rows for which we don't have a category
        frame = frame[frame['category_code'].notnull()]

        # Since we're working with chunks, there's the possibility that
        # We've fallen in a chunk we're every row didn't have a category
        if not frame.empty:
            frame = subcategories_extractor(frame, to_drop=cols_to_drop)

            # Count the number of sold products for each category and product in the chunk
            frame = frame.groupby(['category', 'product_id'], sort=False).count()
        
            # Append the results
            results = results.append(frame)

        del frame
        gc.collect()

    # To extract the most sold products, we need to sum over the category and product
    # then, we need to sort for each group of the category groupby and take the first 10 elements (head)
    results = results.groupby(['category', 'product_id']).sum()
    results = results.groupby('category', group_keys=False, sort=False).apply(lambda x: x.sort_values(by='event_type', ascending=False).head(10)).reset_index()
    
    if cat == None:
        return results
    
    gc.collect()
    return results[results['category'] == cat]


# [RQ3] Functions

# 3.a

def avg_price_cat(df_paths, category):
    """Plot the average price of the products sold by the brands within a category.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to plot the statistics.
        category (string): Category for which we want to plot the statistics.
    """
    df = iter_all_dfs(df_paths, ['event_type', 'category_code', 'brand', 'price'])

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']
    
    # Function we will use to aggregate the results
    # To compute the mean, since we're working with chunks, we need to
    # keep both the sum and the count
    def f(x):
        d = {}
        d['price_sum'] = x['price'].sum()
        d['price_count'] = x['price'].count()
        return pd.Series(d, index=['price_sum', 'price_count'])
    

    results = pd.DataFrame()

    for frame in df:
        frame = only_purchases(frame)
        frame = frame[frame['category_code'].notnull()]
        if not frame.empty:
            frame = subcategories_extractor(frame, to_drop=cols_to_drop)

            # Eventually, we are going to compute an average. To do so, we need to
            # keep both the sum and the count for each chunk, since if the count changes
            # each mean has a different weight
            frame = frame.loc[frame['category'] == category].groupby('brand', sort=False).apply(f)
        
            results = results.append(frame)

        del frame
        gc.collect()

    # Once we've worked every chunk, we can sum the sums and the counts and compute the mean
    results = results.groupby(['brand']).sum()

    # Mean:
    results = (results['price_sum'] / results['price_count'])
    
    # Let's extract some stats before plotting the average price
    sorted = results.sort_values()

    quantiles_perc = [0.25, 0.50, 0.75, 0.95]

    quantiles = [sorted.quantile(quantiles_perc[i]) for i in range(len(quantiles_perc))]

    quantiles_x = [np.where(sorted == sorted[sorted <= quantiles[i]].max()) for i in range(len(quantiles))]

    quantile_colors = ['forestgreen', 'sandybrown', 'lightseagreen', 'violet']

    # Plot
    f = plt.figure()

    ax = sorted.plot(figsize=(15, 6), color='royalblue', zorder=3)
    ax.axes.xaxis.set_ticklabels([])
    plt.axhline(y=sorted.mean(), color='red', linestyle='--')

    for i in range(len(quantiles_x)):
        plt.axvline(x=quantiles_x[i], color=quantile_colors[i], linestyle='--')

    plt.grid(color='lightgray', linestyle='-.', zorder=0)

    plt.setp(ax, xlabel='brands', ylabel='average price', title='Average price per brand in category `{}`'.format(category))
    ax.legend(['Average price per brand', 'Mean', '25% Quantile', '50% Quantile', '75% Quantile', '95% Quantile'])
    plt.show()

    gc.collect
    return sorted

def plots_quantile_avg(avg_price_brands):
    
    quantiles_perc = [0.25, 0.50, 0.75, 0.95]

    quantiles = [avg_price_brands.quantile(quantiles_perc[i]) for i in range(len(quantiles_perc))]
    
    
    quantile_df2 = avg_price_brands[avg_price_brands >= quantiles[3]]

    quantile_df1 = avg_price_brands[(avg_price_brands >= quantiles[2]) & (avg_price_brands <= quantiles[3])][:quantile_df2.shape[0]]

    _, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    plots = [quantile_df1, quantile_df2]
    plots_colors = ['orange', 'tomato']

    for i, ax in enumerate(axes.flatten()):
        plots[i].plot(kind='bar', edgecolor='black', ax=ax, color=plots_colors[i], zorder=3)        
        ax.set_ylim([0, quantile_df2.max() + 200])
        ax.set_xticklabels(labels=plots[i].index, rotation = 45, ha="right")

        ax.grid(color='lightgray', linestyle='-.', zorder=0)

    plt.show()

# 3.b

def highest_price_brands(df_paths):
    """Compute the brand with the highest average price for each category.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.

    Returns:
        pd.DataFrame: Dataframe containing the category, the highest price brand along with its average price in ascending order.
    """
    df = iter_all_dfs(df_paths, ['category_code', 'brand', 'price'] , chunksize=1000000)

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']

    # To compute the mean in chunks, we need both the sum and the count of the elements    
    def f(x):
        d = {}
        d['price_sum'] = x['price'].sum()
        d['price_count'] = x['price'].count()
        return pd.Series(d, index=['price_sum', 'price_count'])
    
    results = pd.DataFrame()
    for frame in df:
        frame = frame[frame['category_code'].notnull()]

        if not frame.empty:
            frame = subcategories_extractor(frame, to_drop=cols_to_drop)
        
            # Group on the category and brand and extract sum and count
            frame = frame.groupby(['category', 'brand']).apply(f).reset_index()
            
            results = results.append(frame)

        del frame
        gc.collect()
    
    # Once we finished working on the chunks, to extract the mean use the sums and counts summed up
    results = results.groupby(['category', 'brand']).sum()
    results['price_avg'] = results['price_sum'] / results['price_count']

    # We don't need the price_sum and price_count columns anymore since we have the mean
    results = results.drop(columns=['price_sum', 'price_count'])

    # To extract the max for each category, we can use the groupby on the idxmax
    results = results.iloc[results.reset_index().groupby('category').idxmax()['price_avg']].sort_values(by='price_avg')
    
    return results

# [RQ4] functions

# 4.a

def monthly_profit_all_brands(df_paths, brand):
    """Compute profit of all the brands for each month and in particular of a given one.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.
        brand (strin): Brand we want to analyze.

    Returns:
        pd.DataFrame: Dataframe containing the profit for each brand across the monhts.
        pd.DataFrame: Dataframe containing the profit for the given brand across the months.
    """
    # Initialize a big dataframe
    entire_df = pd.DataFrame()

    # We are going to work on each dataframe-month separately
    for month_path in df_paths:
        
        # Read the month-dataframe
        df = pd.read_csv(month_path, usecols=['event_type', 'brand', 'price'], iterator=True, chunksize=100000)
        
        # Iterate through the chunks of the given month-dataframe
        results = pd.DataFrame()
        for frame in df:
            frame = frame[frame['brand'].notnull()]
            if not frame.empty:
                # From the purchases-df, extract the sum of the prices for each brand
                frame = only_purchases(frame)
                frame = frame.groupby('brand').sum()

                # Append the df
                results = results.append(frame)
        
        # Extract the month name using some simple parsing
        month_name = month_path.split('-')[1][:3]

        # Sum again across the groups w.r.t. the brand column
        # This is the results for each month
        results = results.groupby('brand').sum().rename(columns={'price': month_name})

        # Append to the big dataframe the results for each month
        entire_df = pd.concat([entire_df, results], axis=1).fillna(0)
    
    # Return both the entire dataframe with all the brands and the one with only the selected one
    return entire_df, entire_df[entire_df.index == brand]

def compare_profits(all_brands_df):
    """Prints general statistics about the profits computed with the monthly_profit_all_brands function.

    Args:
        all_brands_df (pd.DataFrame): Dataframe containing the stats for all the brands in the dataframes.
    """
    print('Mean of the brand profits along all the months considered: \n')
    print(all_brands_df.mean(axis=1), '\n')

    print('Average of the profits along all the brands (all months): \n')
    print(all_brands_df.mean(axis=1).mean(), '\n')

    print('Max of the profits along all the brands (all months): \n')
    print(all_brands_df.mean(axis=1).max(), '\n')

    print('Min of the profits along all the brands (all months): \n')
    print(all_brands_df.mean(axis=1).min(), '\n')

# 4.b

def top_losses(all_brands, num_worst=3):
    """Compute the top brands that suffered the biggest losses in earnings between one month and the next.

    Args:
        all_brands (pd.DataFrame): Dataframe containing the profit data for all the brands across all the months.
        num_worst (int, optional): Num of worst brands to show. Defaults to 3.
    """
    # Create a copy, in which we are going to put the differences between the months profits
    months_diff_df = all_brands.copy()
    cols = all_brands.columns
    new_cols = []

    # Generate the differences for each month
    for i in range(0, len(cols) - 1):
        month_diff = cols[i] + '-' + cols[i + 1]
        new_cols.append(month_diff)
        months_diff_df[month_diff] = months_diff_df[cols[i]] - months_diff_df[cols[i + 1]]

    # Select just the new columns
    months_diff_df = months_diff_df[new_cols]
    
    # Extract the max for each row, that is the month in which each brand suffered the most
    # Along with its index
    max_rows = pd.DataFrame(data=[months_diff_df.max(axis=1), months_diff_df.idxmax(axis=1)]).T

    # Using some parsing techniques, we can extrapolate the months where we had the max losses
    max_rows[['first_month', 'second_month']] = max_rows[1].str.split('-', expand=True)

    # Sort these values and take the first num_worst values 
    max_rows = max_rows.sort_values(by=0, ascending=False).drop(columns=[0, 1]).head(num_worst)
    
    # For each one of these, find the first and second month where the brand had its loss
    # and the corresponding percentage lost. Then print these values
    for i in range(num_worst):
        brand = max_rows.index[i]
        month_1 = max_rows['first_month'][i]
        month_2 = max_rows['second_month'][i]
        value_month_1 = all_brands[all_brands.index == brand][month_1].item()
        value_month_2 = all_brands[all_brands.index == brand][month_2].item()
        percentage_lost = 100 - value_month_2 / (value_month_1 / 100)
        print('{} lost {:.2f}% between {} and {}'.format(brand, percentage_lost, month_1, month_2))

# [RQ5] functions

def avg_users(df_paths):
    """For each day of the weeek, plot the hourly average number of visitors the store has.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.

    Returns:
        defaultdict: Dictionary containing the dataframes of the statistics for each weekday.
    """
    df = iter_all_dfs(df_paths, ['event_time', 'user_id'])

    n_weekdays = [0, 0, 0, 0, 0, 0, 0]
    
    # We're going to define a defaultdict containing the data for each weekday
    # This means that by default, its value should be an empty dataframe
    # since we're going to use dataframes to store the data for each weekday
    def def_value():
        return pd.DataFrame()

    week_days = defaultdict(def_value)
    
    # Iterate in the chunks
    for frame in df:
        # First parse the column for the dates
        frame = df_parsed(frame)

        # For each chunk, we need to see which unique weekday there are
        unique_dates = frame.event_time.dt.strftime('%d-%m-%y').unique()
        
        # Update the number of weekdays we're seeing (# mondays, # tuesdays...)
        for date in unique_dates:
            n_weekdays[datetime.strptime(date, "%d-%m-%y").weekday()] += 1

        # Group for the weekday, then count grouping on each hour of that specific weekday
        for _, week_day_df in frame.groupby(frame.event_time.dt.weekday):
            users_num = week_day_df.groupby(week_day_df.event_time.dt.hour).count()
            current_weekday = week_day_df.event_time.iloc[0].strftime('%A')

            # Append the data to the corresponding dataframe in the dictionary
            week_days[current_weekday] = week_days[current_weekday].append(users_num['user_id'])

    # Now we need to compute the averages, for each weekday. For this we use the number of weekdays
    # we computed in the previous calculations
    for day in week_days:
        week_days[day] = week_days[day].T.sum(axis=1)
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

    # Return the data for all the weekdays
    return week_days

# [RQ6] functions

# 6.a

def purch_view(df):
    """Compute the number of purchases and the number of views per product for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations.

    Returns:
        float: Number of purchases in the dataframe.
        float: Number of views in the dataframe.
    """
    views = only_views(df)
    purchases = only_purchases(df)

    n_purchases = purchases.groupby('product_id', sort=False)['event_type'].count().sum().item()
    n_views = views.groupby('product_id', sort=False)['event_type'].count().sum().item()
    return n_purchases, n_views


def conversion_rate(df_paths):
    """Compute the conversion rate (# purchases / # views) of the store.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.

    Returns:
        float: Conversion rate of the store.
    """
    # Load the iterable for all the dataframes in the paths
    df = iter_all_dfs(df_paths, ['event_type', 'product_id'])

    # Initialize the values for purchases and views
    n_purchases = 0
    n_views = 0

    for frame in df:
        # For each chunk, compute the number of purchases and views and sum to the previous
        purchases, views = purch_view(frame)
        n_purchases += purchases
        n_views += views
    
    # Return the conversion rate
    return n_purchases / n_views

# 6.b

def category_conv_rate(df_paths):
    """Plot and return the conversion rate of each category in decreasing order.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics.

    Returns:
        pd.DataFrame: Dataframe containing the conversion rate for each category of the df.
    """
    df = iter_all_dfs(df_paths, ['event_type', 'product_id', 'category_code'])

    cols_to_drop = ['sub_category_1', 'sub_category_2', 'sub_category_3']

    # We're going to work with a defaultdict, which represetn the number of purchases and views
    # For this reason we're using a np array with both zeros, because it's very reliable in terms
    # of broadcasting operations    
    def def_value():
        return np.array([0, 0], dtype=float)

    purchases_and_views = defaultdict(def_value)

    for frame in df:
        frame = frame[frame['category_code'].notnull()]
        if not frame.empty:
            frame = subcategories_extractor(frame, to_drop=cols_to_drop)
            # In each chunk, compute the number of purchases and views for each category
            # and sum them to the previous values in the dictionary
            for category_name, sub_frame in frame.groupby('category', sort=False):
                purchases_and_views[category_name] += purch_view(sub_frame)
    
    # Now we can convert the dictionary to a Pandas dataframe, so that we can use our plotting function easily
    # We're going to use two different columns to store the values, so that it is easy to manage them
    cat_df = pd.DataFrame(purchases_and_views).T.rename(columns={0: 'purch_num', 1: 'views_num'})
    
    # Compute the conversion rate for each category
    cat_df['conversion_rate'] = cat_df['purch_num'] / cat_df['views_num']
    cat_df = cat_df.drop(columns=['purch_num', 'views_num']).sort_values(by=['conversion_rate'], ascending=False)
    
    # Plot
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
    """Compute the percentage of business conducted by a percentage of users.

    Args:
        df_paths (list): List of the paths of the Pandas dataframes we wish to compute the statistics. 
        users_perc (int, optional): Percentage of users to consider. Defaults to 20.

    Returns:
        float: Percentage of business conducted by the given percentage of users.
    """
    df = iter_all_dfs(df_paths, ['event_type', 'price', 'user_id'])
    
    # We decided to use a reduce function here
    # So first we need to define the initial results
    initial_results = {
            'tot_purchases': 0,
            'purchases_for_user': pd.DataFrame(),
            'unique_users': np.array([])
        }

    # Then we need to define the function to use in the reduce
    def accumulate_data(prev, frame):
        purchases = only_purchases(frame)
        return {
            'tot_purchases': prev['tot_purchases'] + purchases['price'].sum(),
            'purchases_for_user': prev['purchases_for_user'].append(purchases.groupby('user_id', sort=False).sum()),
            'unique_users': np.append(prev['unique_users'], purchases['user_id'].unique())
        }

    # At this point it is enough to use the reduce function on the data
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
    return twenty_most / (tot_purchases / 100)


def plot_pareto(df_paths, step=30, color='darkorange'):
    """Plot the trend of the business conducted by chunks of users, with a selected step

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
        step (int, optional): Step of the percentages of users. Defaults to 10.
        color (str, optional): Plot color. Defaults to 'darkorange'.
    """
    x = np.arange(5, 100, step)
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