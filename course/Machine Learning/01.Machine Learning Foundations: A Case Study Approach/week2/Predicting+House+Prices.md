
# Fire up graphlab create
(See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)


```python
import graphlab
```


```python
# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
```

# Load some house sales data

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


```python
sales = graphlab.SFrame('home_data.gl/')
```


```python
sales
```

# Exploring the data for housing sales 

The house price is correlated with the number of square feet of living space.


```python
graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")
```

# Create a simple regression model of sqft_living to price

Split data into training and testing.  
We use seed=0 so that everyone running this notebook gets the same results.  In practice, you may set a random seed (or let GraphLab Create pick a random seed for you).  


```python
train_data,test_data = sales.random_split(.8,seed=0)
```

## Build the regression model using only sqft_living as a feature


```python
sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)
```

# Evaluate the simple model


```python
print test_data['price'].mean()
```


```python
print sqft_model.evaluate(test_data)
```

RMSE of about \$255,170!

# Let's show what our predictions look like

Matplotlib is a Python plotting library that is also useful for plotting.  You can install it with:

'pip install matplotlib'


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')
```

Above:  blue dots are original data, green line is the prediction from the simple regression.

Below: we can view the learned regression coefficients. 


```python
sqft_model.get('coefficients')
```

# Explore other features in the data

To build a more elaborate model, we will explore using more features.


```python
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
```


```python
sales[my_features].show()
```


```python
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')
```

Pull the bar at the bottom to view more of the data.  

98039 is the most expensive zip code.

# Build a regression model with more features


```python
my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)
```


```python
print my_features
```

## Comparing the results of the simple model with adding more features


```python
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)
```

The RMSE goes down from \$255,170 to \$179,508 with more features.

# Apply learned models to predict prices of 3 houses

The first house we will use is considered an "average" house in Seattle. 


```python
house1 = sales[sales['id']=='5309101200']
```


```python
house1
```

<img src="http://info.kingcounty.gov/Assessor/eRealProperty/MediaHandler.aspx?Media=2916871">


```python
print house1['price']
```


```python
print sqft_model.predict(house1)
```


```python
print my_features_model.predict(house1)
```

In this case, the model with more features provides a worse prediction than the simpler model with only 1 feature.  However, on average, the model with more features is better.

## Prediction for a second, fancier house

We will now examine the predictions for a fancier house.


```python
house2 = sales[sales['id']=='1925069082']
```


```python
house2
```

<img src="https://ssl.cdn-redfin.com/photo/1/bigphoto/302/734302_0.jpg">


```python
print sqft_model.predict(house2)
```


```python
print my_features_model.predict(house2)
```

In this case, the model with more features provides a better prediction.  This behavior is expected here, because this house is more differentiated by features that go beyond its square feet of living space, especially the fact that it's a waterfront house. 

## Last house, super fancy

Our last house is a very large one owned by a famous Seattleite.


```python
bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}
```

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Bill_gates%27_house.jpg/2560px-Bill_gates%27_house.jpg">


```python
print my_features_model.predict(graphlab.SFrame(bill_gates))
```

The model predicts a price of over $13M for this house! But we expect the house to cost much more.  (There are very few samples in the dataset of houses that are this fancy, so we don't expect the model to capture a perfect prediction here.)


```python

```
