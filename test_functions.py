from functions import *

from pandas.util.testing import assert_frame_equal
from pandas.core.arrays.categorical import Categorical
from pandas._libs.interval import Interval


def test_clean_data():
    input_data = {f'x{i}': 0.5 for i in range(100)}
    input_data['x5'] = "monday"
    input_data['x12'] = "$5,547.78"
    input_data['x31'] = "germany"
    input_data['x63'] = "36.29%"
    input_data['x80'] = np.nan,
    input_data['x81'] = "October",
    input_data['x82'] = "Female",
    input_df = pd.DataFrame(input_data)

    expected_data = input_data.copy()
    expected_data['x12'] = 5547.78
    expected_data['x63'] = 36.29
    expected = pd.DataFrame(expected_data)

    actual = clean_data(input_df)

    assert_frame_equal(expected, actual)


def test_prep_dataset():
    input_data = [{f'x{i}': 0 for i in range(100)},
                  {f'x{i}': 1 for i in range(100)}]
    input_data[0]['x5'] = "monday"
    input_data[0]['x31'] = "germany"
    input_data[0]['x80'] = np.nan
    input_data[0]['x81'] = "October"
    input_data[0]['x82'] = "Female"
    input_data[0]['y'] = 0
    input_data[1]['x3'] = np.nan
    input_data[1]['x5'] = "tuesday"
    input_data[1]['x31'] = "america"
    input_data[1]['x81'] = "November"
    input_data[1]['x82'] = "Male"
    input_data[1]['y'] = 1
    input_df = pd.DataFrame(input_data)

    # after imputing and standardizing
    expected_data = [{f'x{i}': -1.0 for i in range(100)},
                     {f'x{i}': 1.0 for i in range(100)}]
    expected_data[0]['x3'] = 0.0
    expected_data[0]['x5'] = "monday"
    expected_data[0]['x31'] = "germany"
    expected_data[0]['x80'] = 0.0
    expected_data[0]['x81'] = "October"
    expected_data[0]['x82'] = "Female"
    expected_data[0]['y'] = 0
    expected_data[1]['x3'] = 0.0
    expected_data[1]['x5'] = "tuesday"
    expected_data[1]['x31'] = "america"
    expected_data[1]['x80'] = 0.0
    expected_data[1]['x81'] = "November"
    expected_data[1]['x82'] = "Male"
    expected_data[1]['y'] = 1

    # dummies
    expected_data[0]['x5_tuesday'] = 0
    expected_data[0]['x5_nan'] = 0
    expected_data[0]['x31_germany'] = 1
    expected_data[0]['x31_nan'] = 0
    expected_data[0]['x81_October'] = 1
    expected_data[0]['x81_nan'] = 0
    expected_data[0]['x82_Male'] = 0
    expected_data[0]['x82_nan'] = 0
    expected_data[0]['y'] = 0
    expected_data[1]['x5_tuesday'] = 1
    expected_data[1]['x5_nan'] = 0
    expected_data[1]['x31_germany'] = 0
    expected_data[1]['x31_nan'] = 0
    expected_data[1]['x81_October'] = 0
    expected_data[1]['x81_nan'] = 0
    expected_data[1]['x82_Male'] = 1
    expected_data[1]['x82_nan'] = 0
    expected_data[1]['y'] = 1
    categ_cols = ['x5', 'x31', 'x81', 'x82']
    for col in categ_cols:
        del (expected_data[0][col], expected_data[1][col])
    expected = pd.DataFrame(expected_data).sort_index()
    expected = expected.reindex(sorted(expected.columns), axis=1)
    dummie_cols = [c for c in expected.columns if
                   any(c.startswith(f'{cc}_') for cc in categ_cols)]
    for col in dummie_cols:
        expected[col] = expected[col].astype(np.uint8)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_scaler = StandardScaler()
    actual = prep_dataset(input_df, imputer, std_scaler).sort_index()
    actual = actual.reindex(sorted(actual.columns), axis=1)

    for col in expected.columns:
        print(col)
        assert expected[col].dtype == actual[col].dtype

    assert_frame_equal(expected, actual)


def test_calculate_c_statistic():
    input_data = {'probs': np.arange(0, 1, 0.1),
                  'y': [0] * 5 + [1] * 5}
    input_df = pd.DataFrame(input_data)

    expected = roc_auc_score([0] * 5 + [1] * 5,
                             np.arange(0, 1, 0.1))

    actual = calculate_c_statistic(input_df)

    assert expected == actual


def test_create_probability_bins():
    input_data = {'probs': np.arange(0, 1.01, 0.01)}
    input_df = pd.DataFrame(input_data)

    expected_data = {'probs': np.arange(0, 1.01, 0.01),
                     'prob_bin': Categorical(sorted(
                         [Interval(0, 0.05)] +
                         [Interval(round(i, 2), round(i + 0.05, 2))
                          for j in range(5)
                          for i in np.arange(0, 1, 0.05)]),
                     ordered=True)}
    expected = pd.DataFrame(expected_data).sort_index()
    expected = expected.reindex(sorted(expected.columns), axis=1)

    actual = create_probability_bins(input_df).sort_index()
    actual = actual.reindex(sorted(actual.columns), axis=1)
    actual['prob_bin'] = actual['prob_bin'].apply(
        lambda x: Interval(round(x.left, 2), round(x.right, 2)))

    joined = actual.join(expected, lsuffix='_x', rsuffix='_y')
    print(joined.loc[joined['prob_bin_x'] != joined['prob_bin_y']])
    assert_frame_equal(expected, actual)


def test_get_cutoff():
    input_data = {'probs': np.arange(0, 1.05, 0.05)}
    input_df = pd.DataFrame(input_data)

    expected = 0.75

    actual = get_cutoff(input_df)

    assert expected == actual


def test_make_prediction():
    input_data = {'probs': np.arange(0, 1, 0.1)}
    input_df = pd.DataFrame(input_data)

    expected_data = {'phat': np.arange(0, 1, 0.1),
                     'business_outcome':
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]}
    expected = pd.DataFrame(expected_data)

    actual = make_prediction(input_df)

    assert_frame_equal(expected, actual)
