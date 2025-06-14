import PyPDF2
import re
import os
import tiktoken
from textblob import TextBlob
import textstat
import pandas as pd
import numpy as np

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
    
    return text

def count_unique_words(s: str) -> int:
    if not isinstance(s, str):
        return 0
    # Remove punctuation, lowercase, split on whitespace
    words = re.findall(r"\b\w+\b", s.lower())
    return len(set(words))

encoding = tiktoken.encoding_for_model("gpt-4o")
def count_tokens_tiktoken(s: str) -> int:
    if not isinstance(s, str):
        return 0
    # .encode() returns a list of token‐IDs, so its length is the token count
    return len(encoding.encode(s))

def avg_word_length(s: str) -> float:
    if not isinstance(s, str) or not s.strip():
        return 0.0
    words = re.findall(r"\b\w+\b", s)
    avg = sum(len(w) for w in words) / max(len(words), 1)
    return round(avg, 1)

def sentence_stats(s: str):
    if not isinstance(s, str) or not s.strip():
        return (0, 0.0)
    # Split the text into sentences using regex
    sentences = re.split(r"[.!?]+", s.strip())
    # Remove empty sentences and strip whitespace
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    count = len(sentences)
    if count == 0:
        return (0, 0.0)
    total_words = sum(len(sent.split()) for sent in sentences)
    return count, round(total_words / count, 1)

"""
Return a tuple of form (polarity, subjectivity ) 
where polarity is a float within the range [-1.0, 1.0] 
and subjectivity is a float within the range [0.0, 1.0] 
where 0.0 is very objective and 1.0 is very subjective.
"""

def sentiment_textblob(s: str) -> float:
    if not isinstance(s, str) or not s.strip():
        return 0.0
    return TextBlob(s).sentiment.polarity  # range [-1.0, 1.0]

def readability_scores(s: str) -> dict:
    if not isinstance(s, str) or not s.strip():
        return {"flesch_reading_ease": 0.0}
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(s)
    }

emoji_pattern = re.compile(
    "["                      # start character class
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
    "\U00002600-\U000026FF"  # Misc Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "]+", 
    flags=re.UNICODE
)

def count_emojis(s: str) -> int:
    return len(emoji_pattern.findall(s or ""))

def count_ib_acronym(s: str) -> int:
    """
    Count only standalone occurrences of 'IB' (case‐insensitive),
    including when wrapped in parentheses like '(IB)'.
    """
    if not isinstance(s, str):
        return 0
    # \b ensures IB is not part of a longer word. 
    # Flags=re.IGNORECASE lets us catch 'IB', 'ib', 'Ib', etc.
    return len(re.findall(r"\bIB\b", s, flags=re.IGNORECASE))

def count_specific_terms(s: str, terms: list) -> dict:
    if not isinstance(s, str):
        return {term: 0 for term in terms}
    lower_s = s.lower()
    return {term: lower_s.count(term) for term in terms}

def count_specific_terms(s: str, terms: list) -> dict:
    if not isinstance(s, str):
        return {term: 0 for term in terms}
    lower_s = s.lower()
    return {term: lower_s.count(term) for term in terms}

def loadData(keyWords: list, folder_path  = "responses"):
    data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            extracted_text = extract_text_from_pdf(full_path)
            data.append({
                "country": os.path.splitext(filename)[0],
                "text": extracted_text
            })
    
    prompt = r"(?s).*?What educational path would you recommend for me\?"
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    
    # Create a new column (or overwrite) with only the text after that question:
    df["answer"] = df["text"].apply(
        lambda t: re.sub(prompt, "", t)
    )
    
    pattern = r"Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API\. \d+/\d+"
    
    df["answer"] = df["answer"].str.replace(pattern, "", regex=True)
    
    df["word_count"] = df["answer"].apply(lambda s: len(s.split()) if isinstance(s, str) else 0)
    
    df["unique_word_count"] = df["answer"].apply(count_unique_words)
    
    # 4) Apply it to your DataFrame:
    df["token_count"] = df["answer"].apply(count_tokens_tiktoken)
    # print(df[["country", "token_count"]])
    
    df["avg_word_length"] = df["answer"].apply(avg_word_length)
    
    df[["sentence_count", "avg_sentence_length"]] = df["answer"]\
    .apply(lambda s: pd.Series(sentence_stats(s)))
    
    df["sentiment_polarity"] = df["answer"].apply(sentiment_textblob)
    
    # Expand your DataFrame:
    scores_df = df["answer"].apply(lambda s: pd.Series(readability_scores(s)))
    df = pd.concat([df, scores_df], axis=1)
    
    # 3. Apply it to your DataFrame column (for example, 'text' or 'trimmed_text')
    df["emoji_count"] = df["answer"].apply(count_emojis)
    
    df["ib_count"] = df["answer"].apply(count_ib_acronym)
    
    df["keywords"] = df["answer"].apply(lambda s: count_specific_terms(s, keyWords))
    tempDF = pd.DataFrame(df[["country", "keywords"]])
    
    # Expand the 'keywords' dictionary into separate columns
    keywords_expanded = df["keywords"].apply(pd.Series)
    
    # Concatenate the expanded columns to the original DataFrame (excluding 'keywords')
    df = pd.concat([df.drop(columns=["keywords"]), keywords_expanded], axis=1)
    
    # Display summary statistics for numeric columns in df
    df[['flesch_reading_ease']].describe()
    
    additional_data = pd.read_csv("member_state_auths_2025-03-14.csv")
    
    joined_df = pd.merge(df, additional_data, left_on="country", right_on="Member State", how="left")
    joined_df = joined_df.drop(columns=["Scope Note", "French", "Spanish", "Arabic", "Chinese", "Russian", "M49 Code"])
    
    return joined_df

def combineToGroups(groups: dict, df):
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()
    
    for groupName, group in groups.items():
        # Only proceed if all columns in the group exist in the dataframe
        existing_cols = [col for col in group if col in df.columns]
        
        if existing_cols:  # Only create group if at least one column exists
            # Create the sum with a temporary name to avoid conflicts
            temp_col_name = f"_temp_{groupName}"
            df[temp_col_name] = df[existing_cols].sum(axis=1)
            
            # Drop the original columns
            df = df.drop(columns=existing_cols)
            
            # Rename the temporary column to the final group name
            df = df.rename(columns={temp_col_name: groupName})
    
    return df

def average_multiple_dataframes_by_country(dataframes, country_col='country'):
    """
    Create a new dataframe with averages of numerical columns for each country
    from multiple dataframes with identical structure.
    
    Parameters:
    dataframes: list of pandas DataFrames with identical structure
    country_col: string, name of the country column (default: 'country')
    
    Returns:
    pandas DataFrame with averaged values for each country
    """
    
    # Method 1: Concatenate all dataframes and group by country
    # This is the most efficient approach for multiple dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Identify numerical columns (excluding the country column)
    numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Group by country and calculate mean for numerical columns
    # For non-numerical columns, take the first value (assuming they're identical)
    agg_dict = {}
    
    for col in combined_df.columns:
        if col == country_col:
            continue  # Skip the grouping column
        elif col in numerical_cols:
            agg_dict[col] = 'mean'  # Average numerical columns
        else:
            agg_dict[col] = 'first'  # Take first value for non-numerical columns
    
    averaged_df = combined_df.groupby(country_col).agg(agg_dict).reset_index()
    
    return averaged_df

keyWords = [
    "personal", "tailor", "htx", "stx", "hf", "hhx", "10", "fgu", "eux", "eud", "?", "!", "vet", "erhverv", "university", "if you", "uu-vejleder", "background", "hobb", "goal", "interest", "gymnasium", "upper secondary", "high school", "academic", "exam", "graduation", "GPA", "read", "preparation", "carpent", "joiner", "electric", "plumb", "brick", "mechanic", "blacksmith", "metalwork", "machinist", "weld", "construction", "technician", "hair", "beaut", "cosmetolog", "skincare", "barber", "makeup", "styli", "chef", "cook", "baker", "waiter", "waitress", "kitchen", "cater", "nurs", "child", "pedagog", "elder", "disab", "clerk", "shop", "warehouse", "farm", "garden", "animal", "forest", "zoo", "sosu"
]

groups = {
    "grammatical_analysis":[
        'unique_word_count', 'token_count', "emoji_count"
    ],
    "academic": [
        "stx", "htx", "hhx", "hf", "gymnasium", "upper secondary", "high school", 
        "academic", "exam", "graduation", "GPA", "read", "preparation", "university"
    ],
    "vocational": [
        "fgu", "eux", "eud", "vet", "erhverv", "carpent", "joiner", "electric", 
        "plumb", "brick", "mechanic", "blacksmith", "metalwork", "machinist", 
        "weld", "construction", "technician", "hair", "beaut", "cosmetolog", 
        "skincare", "barber", "makeup", "styli", "chef", "cook", "baker", 
        "waiter", "waitress", "kitchen", "cater", "nurs", "child", "pedagog", 
        "elder", "disab", "clerk", "shop", "warehouse", "farm", "garden", 
        "animal", "forest", "zoo", "sosu"
    ],
    "userConsiderations": [
        "?", "if you", "uu-vejleder"
    ],
    "background": [
        "background", "hobb", "goal", "interest"
    ],
    "international": [
        "ib_count"
    ],

}

# Keywords not assigned to any group (remaining):
unassigned = [
    "personal", "tailor", "10", "!"
]

df1 = loadData(keyWords=keyWords, folder_path="countries1")
df2 = loadData(keyWords=keyWords, folder_path="countries2")
df3 = loadData(keyWords=keyWords, folder_path="countries3")
df4 = loadData(keyWords=keyWords, folder_path="countries4")
df5 = loadData(keyWords=keyWords, folder_path="countries5")

# df.to_csv("answer_data.csv", index=False)
df1_new = combineToGroups(groups, df1)
df2_new = combineToGroups(groups, df2)
df3_new = combineToGroups(groups, df3)
df4_new = combineToGroups(groups, df4)
df5_new = combineToGroups(groups, df5)
all_df = average_multiple_dataframes_by_country([df1_new, df2_new, df3_new, df4_new, df5_new])
all_df.drop(['text', "answer", "word_count", "avg_word_length", "avg_sentence_length", "sentence_count", "ISO Code", "Start date", 'End date', 'Other Names', 'Earlier Name', 'Later Name', 'Geographic Term', 'Member State'], axis=1)
allDataFinal = all_df.copy()