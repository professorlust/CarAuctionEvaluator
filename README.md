# CarAuctionEvaluator
The goal of this project is to build a predictive model to help auto dealers avoid purchasing potentially bad vehicles. The model will be based on a set of 72,000 historical records that contains 32 attributes as well as a unique id for each purchase and a label indicating if the vehicle was a bad buy or not.

## Preliminary Analysis
- Bad Buy Percentage: 12.30%
- Years range from 2001 to 2010. The median year was 2005 and the mean year was 2005.
- Age ranges from 0 to 9 years old. The median age was 4 and the mean age was 4.
- Odometer readings range from 4825 to 115717 miles. The median reading was 73363 and the mean reading was 71502.

## Features

Field|Description
-------|---
*RefID*|Unique (sequential) number assigned to vehicles
*IsBadBuy*|Identifies if the vehicle was a bad buy
*PurchDate*|The Date the vehicle was purchased at Auction
*Auction*|Auction provider at which the vehicle was purchased
*VehYear*|The manufacturer's year of the vehicle
*VehicleAge*|The Years elapsed since the manufacturer's year
*Make*|Vehicle Manufacturer
*Model*|Vehicle Model
*Trim*|Vehicle Trim Level
*SubModel*|Vehicle Sub-model
*Color*|Vehicle Color
*Transmission*|Vehicles transmission type (Automatic, Manual)
*WheelTypeID*|The type id of the vehicle wheel
*WheelType*|The vehicle wheel type description (Alloy, Covers)
*VehOdo*|The vehicles odometer reading
*Nationality*|The Manufacturer's country
*Size*|The size category of the vehicle (Compact, SUV, etc.)
*TopThreeAmericanName*|Identifies if the manufacturer is one of the top three American manufacturers
*MMRAcquisitionAuctionAveragePrice*|Acquisition price for this vehicle in average condition at time of purchase
*MMRAcquisitionAuctionCleanPrice*|Acquisition price for this vehicle in the above Average condition at time of purchase
*MMRAcquisitionRetailAveragePrice*|Acquisition price for this vehicle in the retail market in average condition at time of purchase
*MMRAcquisitonRetailCleanPrice*|Acquisition price for this vehicle in the retail market in above average condition at time of purchase
*MMRCurrentAuctionAveragePrice*|Acquisition price for this vehicle in average condition as of current day
*MMRCurrentAuctionCleanPrice*|Acquisition price for this vehicle in the above condition as of current day
*MMRCurrentRetailAveragePrice*|Acquisition price for this vehicle in the retail market in average condition as of current day
*MMRCurrentRetailCleanPrice*|Acquisition price for this vehicle in the retail market in above average condition as of current day
*PRIMEUNIT*|Identifies if the vehicle would have a higher demand than a standard purchase
*AcquisitionType*|Identifies how the vehicle was acquired (Auction buy, trade in, etc)
*AUCGUART*|The level guarantee provided by auction for the vehicle (Green light - Guaranteed/arbitratable, Yellow Light - caution/issue, red light - sold as is)
*KickDate*|Date the vehicle was kicked back to the auction
*BYRNO*|Unique number assigned to the buyer that purchased the vehicle
*VNZIP*|Zipcode where the car was purchased
*VNST*|State where the the car was purchased
*VehBCost*|Acquisition cost paid for the vehicle at time of purchase
*IsOnlineSale*|Identifies if the vehicle was originally purchased online
*WarrantyCost*|Warranty price (term=36month  and millage=36K)

## Dependencies
- [Numpy](http://www.numpy.org)