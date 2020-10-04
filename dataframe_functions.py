def to_money(val):
    '''
    INPUT:
    str formatted $1,000.00 indicating monetary value 
    
    OUTPUT:
    float of monetary value
    '''
    val = val[1:len(val)-3]
    val = val.replace(',', '')
    return(float(val))

def col_to_money(colname, df):
    '''
    INPUT:
    colname to be converted, dataframe that hosts column 
    
    OUTPUT:
    the column, in a float format
    '''
    return ([to_money(i) if isinstance(i, float) == False else None for i in df[colname]])

def col_from_percent(colname, df):
    '''
    INPUT:
    colname to be converted, dataframe that hosts column 
    
    OUTPUT:
    the column, in a float format
    '''
    return([float(i.replace('%', ''))if isinstance(i, float) == False else None for i in df[colname]])

def flatten_multiindex(df):
    '''
    INPUT:
    df with multiindex (usually result of multiple agg steps)
    
    OUTPUT:
    new colnames
    should be used df.columns = flatten_multiindex(df)
    '''
    return(['_'.join(col).strip() for col in clean_agg.columns.values])

def create_dummy_df(df, cat_cols, dum_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
            
    '''
    new_df = df.copy()
    for col in  cat_cols:
        try:
            dums =  pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dum_na)
            new_df = pd.concat([new_df.drop(col, axis=1),dums], axis=1)
        except:
            continue
    return new_df

def list_to_cols(series):
    '''
    INPUT:
    series consisting of a list 
    
    OUTPUT:
    dataframe where items in the list are columns
    '''
    column_names = []
    for i in series:
        for n in i:
            if n not in column_names:
                column_names.append(n)
    df = pd.DataFrame(columns = column_names)
    for z in column_names:
        df[z] = [1 if z in x else 0 for x in series]
    return(df)

def text_to_cols(colname, df):
    '''
    INPUT:
    str column name and dataframe that contains it 
    
    OUTPUT:
    dataframe with new columns containing str properties, original str column removed
    '''
    fancy_strings = [FancyString(i) for i in df[colname]]
    df[colname + '_adcount'] = [i.ad_count() for i in fancy_strings]
    df[colname + '_wordcount'] = [i.wordcount() for i in fancy_strings]
    df = df.drop(colname, axis = 1)
    return(df)
