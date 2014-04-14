# Run this using R CMD BATCH mkplot.R

# read the table
x <- read.table('data2.table')

# plot the 'on' rates
png('plot-on-rate.png')
plot(x$on)
dev.off()

# plot the 'off' rates
png('plot-off-rate.png')
plot(x$off)
dev.off()

# plot the rates against each other
png('plot-rates.png')
plot(x)
dev.off()

