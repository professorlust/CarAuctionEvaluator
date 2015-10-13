import numpy as np
import csv

# data = numpy.genfromtxt('AuctionCarTrgData.csv', delimiter=',')
reader = csv.reader(open('AuctionCarTrgData.csv', 'rU'), delimiter=',')
data = np.array(list(reader)[1:])
data_matrix = np.asmatrix(data)

num_vehicles = len(data)

num_bad_buys = float(len(filter(lambda x: x == "1", data_matrix[:, 1])))
bad_buy_percentage = num_bad_buys / float(num_vehicles)
print '\nBad Buy Percentage: {:1.2f}%'.format(bad_buy_percentage*100)

year_column = np.array(data_matrix[:, 4]).astype(np.int)
year_counts = np.bincount(year_column[:, 0])
most_popular_year = np.argmax(year_counts)
max_year = np.max(year_column, axis=0)[0]
min_year = np.min(year_column, axis=0)[0]
median_year = np.median(year_column)
mean_year = np.mean(year_column)
print 'Years range from {0} to {1}. The median year was {2:.0f} and the mean year was {3:.0f}. ' \
      'The most popular year is {4}.'.format(min_year, max_year, median_year, mean_year, most_popular_year)

age_column = np.array(data_matrix[:, 5]).astype(np.int)
max_age = np.max(age_column, axis=0)[0]
min_age = np.min(age_column, axis=0)[0]
median_age = np.median(age_column)
mean_age = np.mean(age_column)
print 'Age ranges from {0} to {1} years old. The median age was {2:.0f} and the mean age was {3:.0f}.'.format(min_age, max_age, median_age, mean_age)

make_column = np.array(data_matrix[:, 6])
make_uniques, make_pos = np.unique(make_column, return_inverse=True)
make_counts = np.bincount(make_pos)
make_maxpos = make_counts.argmax()
most_popular_make = make_uniques[make_maxpos]
print 'The most popular make is {0}.'.format(most_popular_make)

color_column = np.array(data_matrix[:, 10])
color_uniques, color_pos = np.unique(color_column, return_inverse=True)
color_counts = np.bincount(color_pos)
color_maxpos = color_counts.argmax()
most_popular_color = color_uniques[color_maxpos]
print 'The most popular color is {0}.'.format(most_popular_color)

odometer_column = np.array(data_matrix[:, 14]).astype(np.int)
max_odo = np.max(odometer_column, axis=0)[0]
min_odo = np.min(odometer_column, axis=0)[0]
median_odo = np.median(odometer_column)
mean_odo = np.mean(odometer_column)
print 'Odometer readings range from {0} to {1} miles. The median reading was {2:.0f} and the mean reading was {3:.0f}.'.format(min_odo, max_odo, median_odo, mean_odo)
