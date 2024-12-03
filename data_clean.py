import os
import pandas as pd

from collections import defaultdict
from label import *

from tqdm import tqdm

# 第一步，检查是否还有非单独映射的变量，即cfps-merge.csv中是否每一列都没有重复值

target_waves = ["2010", "2012", "2014", "2016", "2018", "2020", "2022"]
data_dir = "data"
merge_info = "cfps-merge.csv"

positive_sig = 1
neutral_sig = 1
negative_sig = 0

def binarize_minzu(value, wave):
    if pd.isna(value):
        return value
    return positive_sig if value == "汉族" else negative_sig

def binarize_leader(value, wave):
    if pd.isna(value):
        return value
    return negative_sig if value in [0, 5] else positive_sig

def binarize_onlychild(value, wave):
    if pd.isna(value):
        return value
    return positive_sig if value > 1 else negative_sig

def binarize_leadership(value, wave):
    if pd.isna(value):
        return value
    return positive_sig if value == 1 else negative_sig if value == 3 else neutral_sig

def binarize_five_level(value):
    if pd.isna(value):
        return value
    return positive_sig if value > 3 else neutral_sig if value == 3 else negative_sig

def binarize_three_level(value):
    if pd.isna(value):
        return value
    return positive_sig if value == 1 else neutral_sig if value == 2 else negative_sig

def binarize_seven_level(value):
    if pd.isna(value):
        return value
    return positive_sig if value > 4 else neutral_sig if value == 4 else negative_sig

def binarize_ten_level(value):
    if pd.isna(value):
        return value
    return positive_sig if value > 5 else neutral_sig if value == 5 else negative_sig

def binarize_percp(value, wave):
    if wave == "2010":
        return binarize_five_level(value)
    else:
        return binarize_ten_level(value)

def binarize_text(value):
    if value in DELETE:
        return None
    elif value in POSITIVE:
        return positive_sig
    elif value in NEUTRAL:
        return neutral_sig
    elif value in NEGATIVE:
        return negative_sig

five_level = ["abilityEnglish", "abilityLocal", "abilityMandarian", "abilityMinority", "abilityOther", "familyGiveup", "familyLivewith", "familyProud", "familyRespect", "familySon", "familyVisit", "futureConfidence", "genderFertility", "genderHousework", "genderMarriage", "genderRear", "genderWork", "importAchieve", "importChildren", "importFamily", "importFertility", "importFun", "importHate", "importIntimacy", "importLoniness", "importMissing", "importMoney", "importRear", "infoInternet", "infoMobile", "infoNews", "infoOthers", "infoRadio", "infoTV", "internetImportCommercial", "internetImportEntertain", "internetImportLearn", "internetImportSocial", "internetImportWork", "internetMeet", "internetPhone", "internetReal", "jobWeekend", "languageDialect", "languageEnglish", "languageLocal", "languageMandarian", "languageMinority", "languageOther", "mindsetAchieved1", "mindsetAchieved2", "mindsetAscribed1", "mindsetAscribed2", "rateConfidence", "rateIncome", "rateRelation", "rateSatisfy", "rateStatus", "satisfyEcon", "satisfyHousework", "satisfyMarriage", "schEvalGrade", "schEvalGreat", "schEvalLeader", "schEvalPressure", "schSatifyChinese", "schSatifyEng", "schSatifyHeadt", "schSatifyMath", "schSatifySchool", "status14"]


three_level = ["jobFree"]

seven_level = ["jobNightTime"]

ten_level = ["lifeMeaning", "severityCorruption", "severityEdu", "severityEnv", "severityGap", "severityHousing", "severityInfant", "severityJob", "severityMedical", "severityOld", "severitySecurity", "trustDoctor", "trustGovern", "trustNeighbor", "trustParents", "trustStrangern", "trustUSA", "successEcon", "successEdu", "successEffort", "successFortune", "successRelation", "successStatus", "successTalent"]


special_binary_map = {
    "minzu": binarize_minzu,
    "leader": binarize_leader,
    "leadership": binarize_leadership,
    "leadership14": binarize_leadership,
    "onlychild": binarize_onlychild,
    "percpRelation": binarize_percp,
    "percpWellbeing": binarize_percp,
}


def check_repetition():
    data_path = os.path.join(data_dir, merge_info)
    df = pd.read_csv(data_path)
    df = df[df["keep"] == 1]
    for column in target_waves:
        column_name = f"{column}-label"
        # 检查这一列是否有重复值，如果有，输出重复的value
        # 不考虑缺失值的重复
        column_values = df[column_name].dropna().values
        unique_values = set(column_values)
        if len(column_values) != len(unique_values):
            print(f"Warning: {column_name} has repeated values.")
            for value in unique_values:
                if column_values.tolist().count(value) > 1:
                    print(f"Repeated value: {value}")
        else:
            print(f"{column_name} has no repeated values.")
    
    column_values = df["variable"].dropna().values
    unique_values = set(column_values)
    if len(column_values) != len(unique_values):
        print("Warning: variable has repeated values.")
        for value in unique_values:
            if column_values.tolist().count(value) > 1:
                print(f"Repeated value: {value}")
    else:
        print("variable has no repeated values.")

def print_unnamed_columns(df):
    unnamed_columns = [col for col in df.columns if "Unnamed" in str(col)]
    print(unnamed_columns)

#根据cfps-merge的各年variable名，与对应的variable列名字构建映射，根据映射重命名年份的列名
def rename_columns():
    cur_labels = POSITIVE + NEUTRAL + NEGATIVE
    text_labels = []
    numeric_variables = set()
    column_name_map = defaultdict(dict)
    merge_data_path = os.path.join(data_dir, merge_info)
    merge_info_data = pd.read_csv(merge_data_path)
    merge_info_data = merge_info_data[merge_info_data["keep"] == 1]
    for index, row in merge_info_data.iterrows():
        for wave in target_waves:
            wave_column_name = f"{wave}-label"
            # 判断该行的wave列是否有值
            if pd.notna(row[wave_column_name]):
                column_name_map[wave][row[wave_column_name]] = row["variable"]
    
    all_columns = merge_info_data["variable"].values
    # 增加pid列
    all_columns = ["pid"] + all_columns.tolist()
    
    for wave in target_waves:
        wave_data_path = os.path.join(data_dir, f"{wave}.csv")
        df = pd.read_csv(wave_data_path)
        # for c in df.columns:
        #     if pd.isna(c):
        #         print(c)
        #     if c not in column_name_map[wave]:
        #         print(c)
        # for k, v in column_name_map[wave].items():
        #     if v is None or pd.isna(v):
        #         print("na", k, v)
        df.rename(columns=column_name_map[wave], inplace=True)
        
        # for c in df.columns:
        #     if pd.isna(c):
        #         print(c)

        # 如果column不在all_columns中，则丢弃
        print_unnamed_columns(df)
        for column in df.columns:
            if column not in all_columns:
                if not column.startswith("qf"):
                    print(f"Warning: wave {wave} {column} is not in all_columns.")
                df.drop(column, axis=1, inplace=True)

        new_wave_data_path = os.path.join(data_dir, f"{wave}_rename.csv")
        df.to_csv(new_wave_data_path, index=False)

        # df中所有数据格式不是text的列
        numeric_variables.update(df.select_dtypes(include=["number"]).columns)

        # df中所有数据格式是text的列的取值
        text_columns = df.select_dtypes(include=["object"]).columns
        for column in text_columns:
            if type(column) == str and "qf" in column:
                continue
            # print(df[column])
            # 去除缺失值
            variables = df[column].dropna().unique()
            for var in variables:
                if var in cur_labels:
                    continue
                text_labels.append(f"{wave}-{column}={var}")
    
    with open("text_labels.txt", "w") as f:
        for text_label in text_labels:
            f.write(f"{text_label}\n")
    
    # 排序numeric_variables
    numeric_variables = sorted(list(numeric_variables))
    with open("numeric_variables.txt", "w") as f:
        for numeric_variable in numeric_variables:
            f.write(f"{numeric_variable}\n")


def binarize():
    all_binarize_df = []
    for wave in target_waves:
        wave_data_path = os.path.join(data_dir, f"{wave}_rename.csv")
        df = pd.read_csv(wave_data_path)
        print_unnamed_columns(df)

        for column in df.columns:
            if column == "pid":
                continue
            if column in special_binary_map:
                df[column] = df[column].apply(lambda x: special_binary_map[column](x, wave))
                continue
            # column dtype格式是数字
            if df[column].dtype != "object":
                if column in five_level:
                    df[column] = df[column].apply(binarize_five_level)
                elif column in three_level:
                    df[column] = df[column].apply(binarize_three_level)
                elif column in seven_level:
                    df[column] = df[column].apply(binarize_seven_level)
                elif column in ten_level:
                    df[column] = df[column].apply(binarize_ten_level)
            else:
                df[column] = df[column].apply(binarize_text)
        print_unnamed_columns(df)
        
        binarize_path = os.path.join(data_dir, f"{wave}_binarize_no_neutral.csv")
        df.to_csv(binarize_path, index=False)

        df["year"] = wave
        all_binarize_df.append(df)
    
    all_binarize_df = pd.concat(all_binarize_df)
    print_unnamed_columns(all_binarize_df)
    all_binarize_df.to_csv(os.path.join(data_dir, "all_binarize_no_neutral.csv"), index=False)


def compile_dataset():
    no_neutral_df = pd.read_csv(os.path.join(data_dir, "all_binarize_no_neutral.csv"))
    
    merge_info_data = pd.read_csv(os.path.join(data_dir, merge_info))
    merge_info_data = merge_info_data[merge_info_data["keep"] == 1]
    variable_to_question = {}
    for index, row in merge_info_data.iterrows():
        variable_to_question[row["variable"]] = row["question"]
    
    compiled_data = {
        "pid": [],
        "year": [],
        "variable": [],
        "question": [],
        "answer": []
    }

    for index, row in tqdm(no_neutral_df.iterrows()):
        for column in no_neutral_df.columns:
            if column == "pid" or column == "year":
                continue
            if row[column] is None:
                continue
            compiled_data["pid"].append(row["pid"])
            compiled_data["year"].append(row["year"])
            compiled_data["variable"].append(column)
            compiled_data["question"].append(variable_to_question[column])
            compiled_data["answer"].append(row[column])
    
    compiled_df = pd.DataFrame(compiled_data)
    # 存入parquet
    compiled_df.to_parquet(os.path.join(data_dir, "compiled_data_no_neutral.parquet"), index=False)


if __name__ == '__main__':
    compile_dataset()

    