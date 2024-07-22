from src import DATADIR, OUTPUTDIR
from src.preprocess import (read_data, clean_data, create_time_cols,
                            filter_data, split_train_test, scale_data,
                            make_kernel, make_gp)
from src.plotting import plot_preds


FILENAME = 'fivemin_data.csv'

df = read_data(DATADIR.joinpath(FILENAME))

df = clean_data(df, ['wmo_hr_sun_dur'],
                {'Unnamed: 0': 'datetime',
                 'sol altitude': 'sol_alt',
                 'sol azimuth': 'sol_az'})

df = create_time_cols(df)

filter_bounds = {'days': (1, 1),
                 'month': 3,
                 'year': 2015}


filt_df = filter_data(df,
                      filter_bounds['days'],
                      filter_bounds['month'],
                      filter_bounds['year'])

predictors = ["minutes", "GHE"]

target = ['GHI']

training_ratio = 0.7

X_train, X_test, y_train, y_test = split_train_test(filt_df, predictors, target, training_ratio)

X_train_scaled, y_train_scaled, x_scaler, y_scaler = scale_data(X_train, y_train, mode='standard')

kernel = make_kernel()
gpr = make_gp(kernel, alpha=1., n_restarts_optimizer=20)

gpr.fit(X_train_scaled, y_train_scaled)

y_pred, y_std = gpr.predict(x_scaler.transform(X_test), return_std=True)

plot_preds(filt_df["datetime"].values,
           filt_df[target].values,
           y_pred, y_std, y_scaler,
           X_train.shape[0])
