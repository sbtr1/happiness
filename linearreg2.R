library("lattice")
library("ggplot2")

#Read CSV files:
myDF <- read.csv(file="data_for_regression.csv", header=TRUE, sep=",")

#Linear regression model:
model = lm(happy ~ gdp + life_expectancy + infant_mort + corruption + pressfreedom + crime, data = myDF)
summary(model)

model = lm(happy ~ gdp + life_expectancy + corruption + crime, data = myDF)
summary(model)

#Boxplot by region:
model2 = lm(happy ~ Region, data = myDF)
ggplot(myDF, aes(x = Region, y = happy)) +
  geom_boxplot(fill = "grey80", colour = "blue") +
  scale_x_discrete() + xlab("Region") +
  ylab("Happiness score") +
  theme_bw()


#Residual diagnostic plot:
df2 = data.frame(Fitted = fitted(model),
                       Residuals = resid(model), Region = myDF$Region)
ggplot(df2, aes(Fitted, Residuals, colour = Region)) + geom_point()
ggplot(df2, aes(Fitted, Residuals)) + geom_point()

qqmath( ~ resid(model),
        xlab = "Theoretical Quantiles",
        ylab = "Residuals"
)