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

def views_extractor(df):
    """Returns a slice of the given dataframe with event_type = view

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: slice of the input df with just view instances
    """
    gc.collect
    return df.loc[df.event_type == 'view']

def subcategories_extractor(df):
    """Extracts two columns (categories and subcategories) from the column category_code

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations

    Returns:
        pd.DataFrame: Dataframe with category and sub_category columns
    """
    # Now we can drop the `category_code` column, since we want to
    # split this column in 2, respectively categories and sub categories
    df_wt_cat = df.drop(columns=['category_code'])

    # We create the two columns, first by selecting just the rows with a non-null value as column 
    df_cat_subcat = df[df['category_code'].notnull()]['category_code']

    # And finally by splitting the `column_code` feature in two columns; then just rename
    df_cat_subcat = df_cat_subcat.str.split('.', expand=True).rename(columns={0: 'category', 1: 'sub_category_1', 2: 'sub_category_2', 3: 'sub_category_3'})

    # Once we have the two dataframes, we can merge them back together
    gc.collect()
    return pd.concat([df_wt_cat, df_cat_subcat], axis=1)


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

# [RQ2] Functions

def products_for_category(df):
    """Plot the histogram of the sold products for category (in ascending order)

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """
    # First select the dataframe rows corresponding to a purchase
    purchases = purchases_extractor(df)
    
    # Extract the category and subcategories from the df
    df_with_cats = subcategories_extractor(purchases)

    # To count the number of products sold for each category, we can simply use a groupby on the category
    # Then we have to count and select the `product_id` column
    results = df_with_cats.groupby('category').count()[['product_id']].sort_values(by='product_id', ascending=False)

    # We can then plot the histogram of the sold products for category
    plot_bar(to_plot=results,
             title='Products sold for category',
             xlabel='categories',
             ylabel='products sold',
             color='darkcyan'
             )
    gc.collect()
    return

# 2.a

def most_viewed_subcategories(df, num_subcat=15):
    """Plot the histogram of the viewed products for subcategory (in ascending order)

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations
    """
    # First select the dataframe rows corresponding to a view
    views = views_extractor(df)

    # Extract the category and subcategories from the df
    views_with_cats = subcategories_extractor(views)

    # Count the number of products viewed in each subcategory using a groupby on the first column of subcategories
    results = views_with_cats.groupby('sub_category_1').count()[['event_type']].sort_values(by='event_type', ascending=False).iloc[:num_subcat]

    # We can then plot the histogram of the number of viewed products for sub category
    plot_bar(to_plot=results,
                       title='Views for subcategory',
                       xlabel='subcategories',
                       ylabel='views',
                       color='mediumvioletred'
                       )
    
    gc.collect()
    return

# 2.b
def unique_categories(df):
    """Returns the unique values of the `category` column of the dataframe

    Args:
        df (pd.DataFrame): Dataframe to use for the calculations

    Returns:
        np.array: Unique values of the category column
    """
    purchases = purchases_extractor(df)
    df_with_cats = subcategories_extractor(purchases)

    return df_with_cats['category'].unique()

def best_in_cat(df, cat=None):
    """Returns dataframe containing the most sold products for category.
    If cat is not `None`, then the most sold products for that specific category is returned

    Args:
        df (pd): DataFrame to use for the calculations
        cat (str, optional): Name of the category for which we want the most sold products. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe containing the most sold products for category.
    """
    # Extract purchases
    purchases = purchases_extractor(df)

    # Extract categories
    df_with_cats = subcategories_extractor(purchases)

    # First we have to groupby by the category and product and count the numbers of instances
    df_count = df_with_cats.groupby(['category', 'product_id']).count()

    # We can now groupby again on the category and apply the lambda to sort the results in descending order
    df_bests_in_cat = df_count['event_type'].groupby('category', group_keys=False).apply(lambda x: x.sort_values(ascending=False).head(10)).reset_index()
    
    # If cat == None just return the dataframe containing all the categories
    if cat == None:
        return df_bests_in_cat
    
    # If one category is given, return the dataframe for that particular one
    return df_bests_in_cat[df_bests_in_cat['category'] == cat]


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
    return purchases.groupby('product_id').count()['event_type'].sum() / views.groupby('product_id').count()['event_type'].sum()

def overall_conversion_rate(df):
    """Apply conversion_rate to return the conversion rate of a dataframe

    Args:
        df (pd.DataFrame): Dataframe to use for calculations

    Returns:
        float: Conversion rate of the given dataframe
    """
    df_with_cats = subcategories_extractor(df)
    views = views_extractor(df)
    purchases = purchases_extractor(df)
    return conversion_rate(views, purchases)

# 6.b

def category_conv_rate(df):
    """Return and plot the conversion rate for each category of the dataframe

    Args:
        df (pd.DataFrame): Dataframe to use for calculations

    Returns:
        pd.DataFrame: Dataframe containing the conversion rates for each category
    """
    dict = {}

    df_with_cats = subcategories_extractor(df)

    for _, frame in df_with_cats.groupby('category'):

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
