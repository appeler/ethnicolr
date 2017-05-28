# census data

# Set dir. 
setwd(githubdir)
setwd("ethnicolor/data-raw")

# 2000 and 2010 Census datasets

# Census 2k
# ---------------
# Read in raw data
cs2000 <- read.csv("census_2000.csv")

# Lots of "(S)" in the data --- these are suppressed proportions where count is 1 to 4
# Imputing (S): Split 'remaining' percentage equally between the "(S)" though 0 is reasonable too

# Convert to numeric 
cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")] <- lapply(cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], as.numeric)

# Missing percentage 
cs2000$remaining_perc <- 100 - rowSums(cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], na.rm=T)

# 4649 rows have negative remaining perc., likely due to rounding error
# For these rows, there shouldn't be NAs, but if there are there they should be imputed to 0
# No NAs: summary(is.na(cs2000[cs2000$remaining_perc <= 0, c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")]))

# Total NAs per row 
cs2000$total_nas <- apply(cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], 1, function(x) sum(is.na(x)))

# Imputed number per row 
cs2000$imputed_num <- ifelse(cs2000$total_nas > 0, cs2000$remaining_perc/cs2000$total_nas, 0)

# Impute numbers
cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")] <- lapply(cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], function(x) ifelse(is.na(x), cs2000$imputed_num, x))

# Placebos
# -----------

mean(100 - rowSums(cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], na.rm=T))
#[1] 0.02505364

mean(apply(cs2000[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], 1, function(x) sum(is.na(x))))
#[1] 0

# --------------------------------------

# Census 2010
# -------------------------

# Read in raw data
cs2010 <- read.csv("census_2010.csv")

# Lots of "(S)" in the data --- these are suppressed proportions where count is 1 to 4
# Imputing (S): Split 'remaining' percentage equally between the "(S)" though 0 is reasonable too

# Convert to numeric 
cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")] <- lapply(cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], as.numeric)

# Missing percentage 
cs2010$remaining_perc <- 100 - rowSums(cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], na.rm=T)

# 4649 rows have negative remaining perc., likely due to rounding error
# For these rows, there shouldn't be NAs, but if there are there they should be imputed to 0
# No NAs: summary(is.na(cs2000[cs2000$remaining_perc <= 0, c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")]))

# Total NAs per row 
cs2010$total_nas <- apply(cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], 1, function(x) sum(is.na(x)))

# Imputed number per row 
cs2010$imputed_num <- ifelse(cs2010$total_nas > 0, cs2010$remaining_perc/cs2010$total_nas, 0)

# Impute numbers
cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")] <- lapply(cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], function(x) ifelse(is.na(x), cs2000$imputed_num, x))

# Placebos
# -----------

mean(100 - rowSums(cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], na.rm=T))
#[1] 0.3836069

mean(apply(cs2010[,c("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")], 1, function(x) sum(is.na(x))))
#[1] 0

# --------------------------------------

# Write out the file
cs2000 <- subset(cs2000, select = -c(remaining_perc, total_nas, imputed_num))
cs2010 <- subset(cs2010, select = -c(remaining_perc, total_nas, imputed_num))

devtools::use_data(cs2000, cs2010, overwrite = TRUE, internal = TRUE)
