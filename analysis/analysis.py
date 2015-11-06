import pandas as pd
import sklearn.ensemble
from prettytable import PrettyTable
import operator
from matplotlib import pyplot

matrix = pd.read_csv('data/train.csv')
bad_vehicles = matrix[(matrix.IsBadBuy == 1)]
good_vehicles = matrix[(matrix.IsBadBuy == 0)]

'''
    Let's first clean up the data and transform some columns into categorical data types
'''
matrix['Transmission'] = matrix['Transmission'].map({"AUTO": "AUTO", "MANUAL": "MANUAL", "Manual": "MANUAL"})

for column in ['Size', 'Make', 'Color', 'Model', 'Transmission', 'WheelType']:
    matrix[column] = matrix[column].astype('category')

'''
    Let's first crunch some numbers to get a good idea of the data we are working with
'''


def examine_column(title, column, bad_buy_set, good_buy_set):
    table = PrettyTable([title, "Min", "Max", "Mode", "Median", "Mean"])
    table.align[title] = "l"
    table.padding_width = 1

    for i in range(0, 3):

        if i == 0:
            data_set = column
            title = "All Vehicles"
        elif i == 1:
            data_set = bad_buy_set
            title = "Bad Vehicles"
        else:
            data_set = good_buy_set
            title = "Good Vehicles"

        max = data_set.max()
        min = data_set.min()
        mode = ','.join(map(str, data_set.mode()))
        median = data_set.median()
        mean = data_set.mean()

        table.add_row([title, min, max, mode, median, mean])

    print(table)


def examine_columns(columns):
    for i in columns:
        overall_column = matrix[i]
        bad_column = bad_vehicles[i]
        good_column = good_vehicles[i]

        examine_column(i, overall_column, bad_column, good_column)
        print('\n')


num_vehicles = matrix.shape[0]
num_bad_buys = len(bad_vehicles)
num_good_buys = len(good_vehicles)
bad_buy_percentage = num_bad_buys / float(num_vehicles)
good_buy_percentage = 1.0 - bad_buy_percentage

table = PrettyTable(['', 'Count', 'Percentage'])
table.add_row(['All Vehicles', num_vehicles, '100%'])
table.add_row(['Bad Vehicles', num_bad_buys, '{:1.2f}%'.format(bad_buy_percentage * 100)])
table.add_row(['Good Vehicles', num_good_buys, '{:1.2f}%'.format(good_buy_percentage * 100)])
print(table)
print('\n')

examine_columns(
    ['VehYear', 'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
     'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
     'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'VehBCost',
     'WarrantyCost'])

'''
RESULTS

+---------------+-------+------------+
|               | Count | Percentage |
+---------------+-------+------------+
|  All Vehicles | 72929 |    100%    |
|  Bad Vehicles |  8969 |   12.30%   |
| Good Vehicles | 63960 |   87.70%   |
+---------------+-------+------------+


+---------------+------+------+------+--------+---------------+
| VehYear       | Min  | Max  | Mode | Median |      Mean     |
+---------------+------+------+------+--------+---------------+
| All Vehicles  | 2001 | 2010 | 2006 | 2005.0 | 2005.34319681 |
| Bad Vehicles  | 2001 | 2009 | 2005 | 2005.0 | 2004.60887501 |
| Good Vehicles | 2001 | 2010 | 2006 | 2006.0 | 2005.44616948 |
+---------------+------+------+------+--------+---------------+


+---------------+-----+-----+------+--------+---------------+
| VehicleAge    | Min | Max | Mode | Median |      Mean     |
+---------------+-----+-----+------+--------+---------------+
| All Vehicles  |  0  |  9  |  4   |  4.0   | 4.17688436699 |
| Bad Vehicles  |  1  |  9  |  5   |  5.0   | 4.94101906567 |
| Good Vehicles |  0  |  9  |  4   |  4.0   | 4.06973108193 |
+---------------+-----+-----+------+--------+---------------+


+---------------+------+--------+-------------------------------------------+---------+---------------+
| VehOdo        | Min  |  Max   |                    Mode                   |  Median |      Mean     |
+---------------+------+--------+-------------------------------------------+---------+---------------+
| All Vehicles  | 4825 | 115717 |                75009,77995                | 73363.0 | 71501.6583389 |
| Bad Vehicles  | 4825 | 115717 | 62296,64895,72392,74273,78044,80350,92005 | 76548.0 | 74718.3618018 |
| Good Vehicles | 5368 | 113617 |             71225,75009,79015             | 72882.5 | 71050.5855847 |
+---------------+------+--------+-------------------------------------------+---------+---------------+


+-----------------------------------+-----+---------+------+--------+---------------+
| MMRAcquisitionAuctionAveragePrice | Min |   Max   | Mode | Median |      Mean     |
+-----------------------------------+-----+---------+------+--------+---------------+
| All Vehicles                      | 0.0 | 35722.0 | 0.0  | 6098.0 | 6129.09855852 |
| Bad Vehicles                      | 0.0 | 35722.0 | 0.0  | 5005.5 | 5411.50747101 |
| Good Vehicles                     | 0.0 | 20327.0 | 0.0  | 6241.0 | 6229.74064401 |
+-----------------------------------+-----+---------+------+--------+---------------+


+---------------------------------+-----+---------+------+--------+---------------+
| MMRAcquisitionAuctionCleanPrice | Min |   Max   | Mode | Median |      Mean     |
+---------------------------------+-----+---------+------+--------+---------------+
| All Vehicles                    | 0.0 | 36859.0 | 0.0  | 7304.0 | 7373.85212108 |
| Bad Vehicles                    | 0.0 | 36859.0 | 0.0  | 6203.0 | 6626.06935772 |
| Good Vehicles                   | 0.0 | 24015.0 | 0.0  | 7450.0 |  7478.7285864 |
+---------------------------------+-----+---------+------+--------+---------------+


+----------------------------------+-----+---------+------+--------+---------------+
| MMRAcquisitionRetailAveragePrice | Min |   Max   | Mode | Median |      Mean     |
+----------------------------------+-----+---------+------+--------+---------------+
| All Vehicles                     | 0.0 | 39080.0 | 0.0  | 8446.0 | 8498.25926129 |
| Bad Vehicles                     | 0.0 | 39080.0 | 0.0  | 7456.0 | 7762.27363961 |
| Good Vehicles                    | 0.0 | 24730.0 | 0.0  | 8571.0 | 8601.48117855 |
+----------------------------------+-----+---------+------+--------+---------------+


+-------------------------------+-----+---------+------+--------+---------------+
| MMRAcquisitonRetailCleanPrice | Min |   Max   | Mode | Median |      Mean     |
+-------------------------------+-----+---------+------+--------+---------------+
| All Vehicles                  | 0.0 | 41482.0 | 0.0  | 9790.0 | 9852.18918956 |
| Bad Vehicles                  | 0.0 | 41482.0 | 0.0  | 8827.0 |  9096.5897636 |
| Good Vehicles                 | 0.0 | 27513.0 | 0.0  | 9921.0 |  9958.1619411 |
+-------------------------------+-----+---------+------+--------+---------------+


+-------------------------------+-----+---------+------+--------+---------------+
| MMRCurrentAuctionAveragePrice | Min |   Max   | Mode | Median |      Mean     |
+-------------------------------+-----+---------+------+--------+---------------+
| All Vehicles                  | 0.0 | 35722.0 | 0.0  | 6062.5 |  6132.1417633 |
| Bad Vehicles                  | 0.0 | 35722.0 | 0.0  | 5015.0 | 5423.90019022 |
| Good Vehicles                 | 0.0 | 21837.0 | 0.0  | 6207.0 | 6231.54272343 |
+-------------------------------+-----+---------+------+--------+---------------+


+-----------------------------+-----+---------+------+--------+---------------+
| MMRCurrentAuctionCleanPrice | Min |   Max   | Mode | Median |      Mean     |
+-----------------------------+-----+---------+------+--------+---------------+
| All Vehicles                | 0.0 | 36859.0 | 0.0  | 7313.0 | 7390.77727435 |
| Bad Vehicles                | 0.0 | 36859.0 | 0.0  | 6212.0 | 6645.86583865 |
| Good Vehicles               | 0.0 | 25847.0 | 0.0  | 7450.0 | 7495.32481116 |
+-----------------------------+-----+---------+------+--------+---------------+


+------------------------------+-----+---------+------+--------+---------------+
| MMRCurrentRetailAveragePrice | Min |   Max   | Mode | Median |      Mean     |
+------------------------------+-----+---------+------+--------+---------------+
| All Vehicles                 | 0.0 | 39080.0 | 0.0  | 8729.0 | 8775.22010907 |
| Bad Vehicles                 | 0.0 | 39080.0 | 0.0  | 7624.0 | 7919.58755735 |
| Good Vehicles                | 0.0 | 24084.0 | 0.0  | 8889.0 | 8895.30723809 |
+------------------------------+-----+---------+------+--------+---------------+


+----------------------------+-----+---------+------+---------+---------------+
| MMRCurrentRetailCleanPrice | Min |   Max   | Mode |  Median |      Mean     |
+----------------------------+-----+---------+------+---------+---------------+
| All Vehicles               | 0.0 | 41062.0 | 0.0  | 10103.0 | 10144.8739086 |
| Bad Vehicles               | 0.0 | 41062.0 | 0.0  |  8984.0 | 9260.91596733 |
| Good Vehicles              | 0.0 | 28415.0 | 0.0  | 10268.0 | 10268.9364763 |
+----------------------------+-----+---------+------+---------+---------------+


+---------------+--------+---------+--------+--------+---------------+
| VehBCost      |  Min   |   Max   |  Mode  | Median |      Mean     |
+---------------+--------+---------+--------+--------+---------------+
| All Vehicles  |  1.0   | 45469.0 | 7500.0 | 6700.0 | 6730.95510606 |
| Bad Vehicles  |  1.0   | 45469.0 | 4200.0 | 6000.0 | 6260.17725833 |
| Good Vehicles | 1400.0 | 16345.0 | 7500.0 | 6800.0 | 6796.97146811 |
+---------------+--------+---------+--------+--------+---------------+


+---------------+-----+------+------+--------+---------------+
| WarrantyCost  | Min | Max  | Mode | Median |      Mean     |
+---------------+-----+------+------+--------+---------------+
| All Vehicles  | 462 | 7498 | 920  | 1155.0 | 1276.59667622 |
| Bad Vehicles  | 462 | 6492 | 920  | 1243.0 | 1359.83353774 |
| Good Vehicles | 462 | 7498 | 920  | 1155.0 | 1264.92451532 |
+---------------+-----+------+------+--------+---------------+
'''

'''
    Now, let's do some visual comparisons of good vs. bad vehicles
'''

plotting = matrix[['VehOdo', 'Color', 'VehBCost', 'WarrantyCost']]
plotting['Color'] = plotting['Color'].cat.codes
color = matrix['IsBadBuy'].map({0: "b", 1: "r"})

axes = pd.tools.plotting.scatter_matrix(plotting, color=color)
pyplot.tight_layout()

fig = pyplot.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
fig.savefig('scatter_matrix.png', dpi=100)

plotting = matrix[['Model', 'Transmission', 'VehYear', 'Size']]
plotting['Model'] = plotting['Model'].cat.codes
plotting['Transmission'] = plotting['Transmission'].cat.codes
plotting['Size'] = plotting['Size'].cat.codes
color = matrix['IsBadBuy'].map({0: "b", 1: "r"})

axes = pd.tools.plotting.scatter_matrix(plotting, color=color)
pyplot.tight_layout()

fig = pyplot.gcf()
fig.set_size_inches(18.5, 10.5, forward=True)
fig.savefig('scatter_matrix_2.png', dpi=100)

'''
Let's perform an analysis of the data to determine which attributes are the most important. To accomplish this
task, I will use ensemble methods. I have included the results from an extra-trees classifier as well as a random
forest classifier. Both classifiers produced very similar results, with the top 3 attributes in both being the
vehicle's odometer reading, the acquisition cost at the time of purchase, and the color of the vehicle. These three
attributes represent a collective 48% importance on determining whether a vehicle is a bad buy or not.
'''


# select only the columns that have enough info to be processed
selected_columns = matrix[['VehYear', 'VehicleAge', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission',
                           'WheelType', 'VehOdo', 'Nationality', 'Size', 'TopThreeAmericanName', 'VehBCost',
                           'IsOnlineSale', 'WarrantyCost']]

# transform categorical attributes into numerical attributes
dummy_matrix = pd.get_dummies(selected_columns)

columns = dummy_matrix.columns.values

# determine most important features
clf = sklearn.ensemble.RandomForestClassifier()  # ExtraTreesClassifier, RandomForestClassifier
important_dummy_matrix = clf.fit(dummy_matrix, matrix['IsBadBuy']).transform(dummy_matrix)

table = PrettyTable(['Attribute', 'Importance (Asc)'])

importances = {}

for i in range(0, len(clf.feature_importances_)):
    column = columns[i]
    importance = clf.feature_importances_[i]

    if '_' in column:
        key = column[0:column.index('_')]
        if key in importances:
            importances[key] += importance
        else:
            importances[key] = importance
    else:
        importances[column] = importance

sorted_importances = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)
values = []
labels = []

for i in range(0, len(sorted_importances)):
    item = sorted_importances[i]
    table.add_row([item[0], item[1]])
    labels.append(item[0])
    values.append(item[1])

print table

pyplot.pie(values, labels=labels)
pyplot.show()

'''
RESULTS

ExtraTreesClassifier
+----------------------+------------------+
|      Attribute       | Importance (Asc) |
+----------------------+------------------+
|        VehOdo        |  0.182264478929  |
|       VehBCost       |  0.179218171237  |
|        Color         |  0.139399453909  |
|      WheelType       |  0.10208820982   |
|     WarrantyCost     | 0.0927523455499  |
|        Model         | 0.0788157018825  |
|       SubModel       | 0.0621972704094  |
|      VehicleAge      | 0.0521127848468  |
|       VehYear        | 0.0439468593445  |
|         Trim         | 0.0268830018191  |
|         Size         | 0.0119000548035  |
|         Make         | 0.00924826780076 |
|     Transmission     | 0.00802005690298 |
|     IsOnlineSale     | 0.00691080077071 |
| TopThreeAmericanName | 0.00256594168079 |
|     Nationality      | 0.00167660029402 |
+----------------------+------------------+

RandomForestClassifier
+----------------------+------------------+
|      Attribute       | Importance (Asc) |
+----------------------+------------------+
|       VehBCost       |  0.179502495658  |
|        VehOdo        |  0.178619492878  |
|        Color         |  0.132193374644  |
|        Model         | 0.0956654477779  |
|      WheelType       | 0.0849051996143  |
|     WarrantyCost     | 0.0804615991005  |
|       SubModel       |  0.077376891251  |
|      VehicleAge      | 0.0439307748765  |
|       VehYear        | 0.0399459284385  |
|         Trim         | 0.0347831677335  |
|         Size         | 0.0164380862182  |
|         Make         | 0.0148917076996  |
|     Transmission     | 0.00774149791537 |
|     IsOnlineSale     | 0.00593632755157 |
| TopThreeAmericanName | 0.00533442347869 |
|     Nationality      | 0.00227358516498 |
+----------------------+------------------+

'''
