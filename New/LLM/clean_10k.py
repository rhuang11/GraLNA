import datasets

# Load the entire dataset
Dataset = datasets.load_dataset("eloukas/edgar-corpus", "full")
datasets.Dataset.to_csv("~/GraLNA/New/LLM/annualreport/annualreportsdataset.csv")

