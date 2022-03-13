from datetime import datetime
from json import dump

import numpy as np
import pandas as pd

state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY"
}

covid_states_data = pd.read_csv("./datasets/us-states.csv",
                                dtype={"fips": str})
covid_counties_data = pd.read_csv("./datasets/us-counties.csv",
                                  dtype={"fips": str})
vaccine_states_data = pd.read_csv("./datasets/us-states-vaccines.csv",
                                  dtype={"FIPS": str})
vaccine_counties_data = pd.read_csv("./datasets/us-counties-vaccines.csv",
                                    dtype={"FIPS": str})

# Rename Columns
covid_states_data.columns = map(str.lower, covid_states_data.columns)
covid_counties_data.columns = map(str.lower, covid_counties_data.columns)
vaccine_states_data.columns = map(str.lower, vaccine_states_data.columns)
vaccine_counties_data.columns = map(str.lower, vaccine_counties_data.columns)

# Add Abbreviation
covid_states_data = covid_states_data[covid_states_data["state"].isin(
    list(state_abbreviations.keys()))]
covid_states_data["location"] = covid_states_data["state"].map(
    lambda x: state_abbreviations[x] if x in state_abbreviations else x)

# Modify Date
covid_states_data["date"] = pd.to_datetime(covid_states_data["date"],
                                           format="%Y-%m-%d")
covid_states_data["month"] = covid_states_data["date"].dt.month
covid_states_data["year"] = covid_states_data["date"].dt.year

covid_counties_data["date"] = pd.to_datetime(covid_counties_data["date"],
                                             format="%Y-%m-%d")
covid_counties_data["month"] = covid_counties_data["date"].dt.month
covid_counties_data["year"] = covid_counties_data["date"].dt.year

vaccine_states_data["date"] = pd.to_datetime(vaccine_states_data["date"],
                                             format="%m/%d/%Y")
vaccine_states_data["month"] = vaccine_states_data["date"].dt.month
vaccine_states_data["year"] = vaccine_states_data["date"].dt.year

vaccine_counties_data["date"] = pd.to_datetime(vaccine_counties_data["date"],
                                               format="%m/%d/%Y")
vaccine_counties_data["month"] = vaccine_counties_data["date"].dt.month
vaccine_counties_data["year"] = vaccine_counties_data["date"].dt.year

# Modify FIPS
covid_counties_data = covid_counties_data[
    covid_counties_data["county"] != "Unknown"]
covid_counties_data = covid_counties_data[~covid_counties_data["fips"].isnull(
)]

vaccine_counties_data = vaccine_counties_data[
    vaccine_counties_data["fips"] != "UNK"]

# Remove Columns
vaccine_states_data.drop(columns=map(str.lower, [
    "Distributed",
    "Distributed_Janssen",
    "Distributed_Moderna",
    "Distributed_Pfizer",
    "Distributed_Unk_Manuf",
    "Dist_Per_100K",
    "Distributed_Per_100k_12Plus",
    "Distributed_Per_100k_18Plus",
    "Distributed_Per_100k_65Plus",
    "Administered_12Plus",
    "Administered_18Plus",
    "Administered_65Plus",
    "Administered_Janssen",
    "Administered_Moderna",
    "Administered_Pfizer",
    "Administered_Unk_Manuf",
    "Admin_Per_100k_12Plus",
    "Admin_Per_100k_18Plus",
    "Admin_Per_100k_65Plus",
    "Administered_Dose1_Recip_12Plus",
    "Administered_Dose1_Recip_12PlusPop_Pct",
    "Administered_Dose1_Recip_18Plus",
    "Administered_Dose1_Recip_18PlusPop_Pct",
    "Administered_Dose1_Recip_65Plus",
    "Administered_Dose1_Recip_65PlusPop_Pct",
    "Series_Complete_12Plus",
    "Series_Complete_12PlusPop_Pct",
    "Series_Complete_18Plus",
    "Series_Complete_18PlusPop_Pct",
    "Series_Complete_65Plus",
    "Series_Complete_65PlusPop_Pct",
    "Series_Complete_Janssen",
    "Series_Complete_Moderna",
    "Series_Complete_Pfizer",
    "Series_Complete_Unk_Manuf",
    "Series_Complete_Janssen_12Plus",
    "Series_Complete_Moderna_12Plus",
    "Series_Complete_Pfizer_12Plus",
    "Series_Complete_Unk_Manuf_12Plus",
    "Series_Complete_Janssen_18Plus",
    "Series_Complete_Moderna_18Plus",
    "Series_Complete_Pfizer_18Plus",
    "Series_Complete_Unk_Manuf_18Plus",
    "Series_Complete_Janssen_65Plus",
    "Series_Complete_Moderna_65Plus",
    "Series_Complete_Pfizer_65Plus",
    "Series_Complete_Unk_Manuf_65Plus",
    "Additional_Doses_18Plus",
    "Additional_Doses_18Plus_Vax_Pct",
    "Additional_Doses_50Plus",
    "Additional_Doses_50Plus_Vax_Pct",
    "Additional_Doses_65Plus",
    "Additional_Doses_65Plus_Vax_Pct",
    "Additional_Doses_Moderna",
    "Additional_Doses_Pfizer",
    "Additional_Doses_Janssen",
    "Additional_Doses_Unk_Manuf",
]),
                         inplace=True)
vaccine_counties_data.drop(columns=map(str.lower, [
    "Series_Complete_12Plus",
    "Series_Complete_12PlusPop_Pct",
    "Series_Complete_18Plus",
    "Series_Complete_18PlusPop_Pct",
    "Series_Complete_65Plus",
    "Series_Complete_65PlusPop_Pct",
    "Completeness_pct",
    "Administered_Dose1_Pop_Pct",
    "Administered_Dose1_Recip_12Plus",
    "Administered_Dose1_Recip_12PlusPop_Pct",
    "Administered_Dose1_Recip_18Plus",
    "Administered_Dose1_Recip_18PlusPop_Pct",
    "Administered_Dose1_Recip_65Plus",
    "Administered_Dose1_Recip_65PlusPop_Pct",
    "SVI_CTGY",
    "Series_Complete_Pop_Pct_SVI",
    "Series_Complete_12PlusPop_Pct_SVI",
    "Series_Complete_18PlusPop_Pct_SVI",
    "Series_Complete_65PlusPop_Pct_SVI",
    "Metro_status",
    "Series_Complete_Pop_Pct_UR_Equity",
    "Series_Complete_12PlusPop_Pct_UR_Equity",
    "Series_Complete_18PlusPop_Pct_UR_Equity",
    "Series_Complete_65PlusPop_Pct_UR_Equity",
]),
                           inplace=True)

# Keep Most Recent Week Per Month Data
covid_states_data.sort_values(by="date", inplace=True)
covid_states_data.drop_duplicates(subset=["fips", "month", "year"],
                                  inplace=True)
covid_states_data = covid_states_data[
    covid_states_data["date"] >= datetime(month=3, day=1, year=2020)]
covid_states_data.drop(columns=["month", "year"], inplace=True)

vaccine_states_data.sort_values(by="date", inplace=True)
vaccine_states_data.drop_duplicates(subset=["location", "month", "year"],
                                    inplace=True)
vaccine_states_data = vaccine_states_data[
    vaccine_states_data["date"] >= datetime(month=3, day=1, year=2020)]
vaccine_states_data.drop(columns=["month", "year"], inplace=True)

covid_counties_data.sort_values(by="date", inplace=True)
covid_counties_data.drop_duplicates(subset=["fips", "month", "year"],
                                    inplace=True)
covid_counties_data = covid_counties_data[
    covid_counties_data["date"] >= datetime(month=3, day=1, year=2020)]
covid_counties_data.drop(columns=["month", "year"], inplace=True)

vaccine_counties_data.sort_values(by="date", inplace=True)
vaccine_counties_data.drop_duplicates(subset=["fips", "month", "year"],
                                      inplace=True)
vaccine_counties_data = vaccine_counties_data[
    vaccine_counties_data["date"] >= datetime(month=3, day=1, year=2020)]
vaccine_counties_data.drop(columns=["month", "year"], inplace=True)

# Merge Datasets
processed_states = pd.merge(covid_states_data,
                            vaccine_states_data,
                            on=["date", "location"],
                            how="left")
processed_counties = pd.merge(covid_counties_data,
                              vaccine_counties_data,
                              on=["date", "fips"],
                              how="left")

processed_states["admin_per_100k"] = processed_states["admin_per_100k"].div(100)
processed_states["administered_dose1_pop_pct"] = processed_states["administered_dose1_pop_pct"].div(100)
processed_states["series_complete_pop_pct"] = processed_states["series_complete_pop_pct"].div(100)

processed_counties["series_complete_pop_pct"] = processed_counties["series_complete_pop_pct"].div(100)

# Export
processed_states.to_csv("./datasets/processed_states.csv", index=False)
processed_counties.to_csv("./datasets/processed_counties.csv", index=False)

# Data by Time
processed_states.fillna(np.nan, inplace=True)
processed_states.replace([np.nan], [None], inplace=True)
processed_counties.fillna(np.nan, inplace=True)
processed_counties.replace([np.nan], [None], inplace=True)

states_by_time = {}
counties_by_time = {}
us_by_time = {}
us_by_time_lst = []

for row in processed_states.to_dict("records"):
    month = row["date"].month - 3
    year = 12 if row["date"].year == 2021 else 0
    row["date"] = datetime.strftime(row["date"], "%m/%d/%Y")
    states_by_time[month + year] = [row] + states_by_time.get(month + year, [])

for k, v in states_by_time.items():
    n_v = {}
    new_d = {}
    counter = 0

    for i in v:
        n_v[i["fips"]] = i
        i_len = len(i.items())
        i.pop("state")
        i.pop("fips")
        i.pop("location")

        for kk, vv in i.items():
            if new_d.get("date") == None:
                new_d[kk] = k
            if new_d.get(kk) == None:
                new_d[kk] = vv 
            else:
                if kk not in new_d:
                    new_d[kk] = vv
                if vv != None and kk != "date":
                    if "100k" in kk or "pct" in kk:
                        new_d[kk] = ((new_d[kk] * counter) + vv) / (counter + 1)
                        counter += 1
                    else:
                        new_d[kk] += vv

        i.pop("date")

    states_by_time[k] = n_v
    us_by_time[k] = new_d
    us_by_time_lst.append(new_d)

for row in processed_counties.to_dict("records"):
    month = row["date"].month - 3
    year = 12 if row["date"].year == 2021 else 0
    row["date"] = datetime.strftime(row["date"], "%m/%d/%Y")
    counties_by_time[month +
                     year] = [row] + counties_by_time.get(month + year, [])

for k, v in counties_by_time.items():
    n_v = {}

    for i in v:
        n_v[i["fips"]] = i

    counties_by_time[k] = n_v

with open("./datasets/processed_states.json", "w") as f:
    dump(states_by_time, f)

with open("./datasets/processed_counties.json", "w") as f:
    dump(counties_by_time, f)

with open("./datasets/processed_us.json", "w") as f:
    dump(us_by_time, f)

pd.DataFrame(us_by_time_lst).to_csv("./datasets/processed_us.csv", index=False)