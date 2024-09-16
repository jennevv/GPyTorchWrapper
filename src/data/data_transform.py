import pandas as pd


def transform_data(x: pd.DataFrame, transformer: object, columns: list=None) -> pd.DataFrame | tuple[pd.DataFrame, object]:
    """
    Transform the input data using the selected transformer

    Parameters
    -----------
    x : pd.DataFrame
        The input data
    transformer : object
                The selected transformer
    columns : list, optional
            The columns on which the transformer has to operate

    Returns
    --------
    x : pd.DataFrame
        The transformed input data
    transformer : object
                The fitted transformer
    """
    # scikit-learn transformers have this attribute
    # check to see if the transformer already has been fitted
    def is_fit_called(obj):
        return hasattr(obj, "n_features_in_")

    if not isinstance(x, pd.DataFrame) and not isinstance(x, pd.Series):
        raise NotImplementedError(f'{type(x)} is not a pandas DataFrame or Series.')

    if columns is not None:
        if all(isinstance(column, str) for column in columns):
            columns_to_transform = columns
        elif all(isinstance(column, int) for column in columns):
            columns_to_transform = x.columns[columns]
        else:
            raise NotImplementedError('All items in the columns list should be the same type (int or string).')

        X_transformed = x[columns_to_transform].copy()

        if is_fit_called(transformer):
            X_transformed = transformer.transform(X_transformed.values)
            x.loc[:, columns_to_transform] = X_transformed
            return x
        else:
            X_transformed = transformer.fit_transform(X_transformed.values)
            x.loc[:, columns_to_transform] = X_transformed
            return x, transformer

    else:
        if is_fit_called(transformer):
            x = pd.DataFrame(transformer.transform(x.values))
            return x
        else:
            x = pd.DataFrame(transformer.fit_transform(x.values))
            return x, transformer
