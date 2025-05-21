import pandas as pd


modality = [['mri', 't1-weighted'],['mri', 't2-weighted'], 'mri', 'x-ray', 'computed tomography']

# for xray i removed lungs
domain = ['chest','lung','lungs' 'brain','abdominal', 'coronal', 'artery', 'lobe', 'tumor','liver' ]




def append_prompt_count_row(df):
    """
    Appends a new row at the bottom of the DataFrame showing the count of non-empty values per column.
    """
    count_row = df.count().rename("# prompts")  # count non-null entries in each column
    df_with_count = df.copy()
    df_with_count.loc["# prompts"] = count_row
    return df_with_count




def extract_data():
    # Read the CSV file, the last column is the caption
    reader_obj = pd.read_csv('ROCO-dataset/radiologytraindata.csv', usecols=['caption'])
    reader_obj = reader_obj.values.tolist()

    return reader_obj


def extract_words(reader_obj):
    # Create an empty dictionary
    d = dict()

    # Loop through each line of the file
    for line in reader_obj:
        # Remove the leading spaces and newline character and lowercase to avoid case mismatch
        line = line[0].strip().lower()

        # Split the line into words
        words = line.split(" ")

        # Iterate over each word in line
        for word in words:
            # Check if the word is already in dictionary
            if word in d:
                # Increment count of word by 1
                d[word] = d[word] + 1
            else:
                # Add the word to dictionary with count 1
                d[word] = 1

    return d

def structure_data_by_modality_or_domain(reader_obj, modalities):
    """
    Groups captions by matching keywords or keyword lists (AND logic).
    Each modality entry can be a string or a list of strings.
    """
    # Convert modalities to dict: key = label, value = list of keywords
    modality_map = {}
    for mod in modalities:
        if isinstance(mod, str):
            modality_map[mod] = [mod]
        elif isinstance(mod, list):
            key = " + ".join(mod)  # Composite label
            modality_map[key] = mod

    # Initialize structured dictionary
    structured_data = {label: [] for label in modality_map}

    for line in reader_obj:
        caption = line[0].strip().lower()
        for label, keywords in modality_map.items():
            if all(kw in caption for kw in keywords):
                structured_data[label].append(caption)
                break  # Assign only once

    # Convert to DataFrame
    df = pd.DataFrame(dict([(label, pd.Series(captions)) for label, captions in structured_data.items()]))
    df = append_prompt_count_row(df)
    return df


def intersect_column_with_all_rows(df1: pd.DataFrame, col1: str, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Compares df1[col1] with all rows in each column of df2.
    If a value in df1[col1] is found anywhere in df2[colX], it's retained; else None.
    """
    result = pd.DataFrame()
    col1_vals = df1[col1]

    for col2 in df2.columns:
        col2_vals_set = set(df2[col2].dropna())  # for faster lookup, skip NaNs
        result[f"{col1} and {col2}"] = [
            val if val in col2_vals_set else None
            for val in col1_vals
        ]

    result = append_prompt_count_row(result)

    return result







if __name__ == "__main__":
    reader_object = extract_data()
    # Append the count row to the CSV file

###############################CREATE MODALITY AND DOMAIN TABLES#################################
    df_modalities = structure_data_by_modality_or_domain(reader_object, modality)
    #df_modalities.to_csv('modality/structured_data_by_modality.csv', index=False)

    df_domains = structure_data_by_modality_or_domain(reader_object, domain)
    #df_domains.to_csv('domain/structured_data_by_domain.csv', index=False)



#####################################INTERSECTION################################################
    # pass the column name for which you want to find the intersection
    intersect_column_with_all_rows(df_modalities, 'mri + t2-weighted', df_domains).dropna(how="all").to_csv(
        'intersection_data/intersection_mri_t2weighted.csv', index=False)





    ####################################EXTRACT WORD COUNT######################################
    #d = extract_words(reader_object)
    #d_sorted = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    # create the table csv
    #df = pd.DataFrame(list(d_sorted.items()), columns=['Word', 'Count'])
    #df.to_csv('word_count.csv', index=False)




