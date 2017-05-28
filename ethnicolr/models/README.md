## General LSTM Model

* Concatenate Last name and first name (in the full name model) and capitalize first character of all the words.

* Split the name into two character chunks (bi-chars) e.g. Smith ==> Sm, mi, it, and th.

* Remove infrequent bi-chars (occurring less than 3 times in the data) and very frequent bi-chars (occurring more than 30% in the dataset).

* Sort by frequency and build up the words list (bi-chars)

* Build X as the index of bi-chars in the words list.

* Pad the sequences in X so that they are the same size. 20 for the last name only model and 25 for the full name model.

* Split into train and test: 80/20 and do out of sample validation

* Train the model with LSTM.

### Census Data 

The Census Bureau provides data on the racial distribution of last names. The dataset that it issues aggregates data for each last name and provides percentage of people with the last name who are Black, White, Asian, Hispanic, etc. Given some names are more common than others (Smith is the last name of 2,376,206 Americans), and given our interest in modeling the population distribution, we take a weighted random sample from this data with weight = how common the last name is in the population or count/total_count. Next, we assign race to name roughly in proportion to how the name is distributed across the racial groups. We assign floor(pctwhite) as whites, floor(pctblacks) as blacks etc. And we lose the one or two or few observations as we are using floor. We use this as the final dataset and apply the model to it.
