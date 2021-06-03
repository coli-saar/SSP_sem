"""
Given the .csv file with scores from experiment runs, find the experiment with the highest performance.

Which scores are considered depends on whether the models were parsing events, participants or both.

This is the full list of column names in the logs_*.csv

index,batch_size,lr,l2,clip,dropout,tagger_dim,corpus_embedding_dim,training_accuracy,training_loss,training_micro_F1,training_macro_F1,training_events_ma_F1,training_events_mi_F1,training_participants_ma_F1,training_participants_mi_F1,validation_accuracy,validation_loss,validation_micro_F1,validation_macro_F1,validation_events_ma_F1,validation_events_mi_F1,validation_participants_ma_F1,validation_participants_mi_F1,test_accuracy,test_loss,test_micro_F1,test_macro_F1,test_events_ma_F1,test_events_mi_F1,test_participants_ma_F1,test_participants_mi_F1,best_epoch,time_consumed(hrs)


"""
import argparse
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="path to the csv file with results")
args = parser.parse_args()

def compare_runs(fname):
	rows = []
	best_avg, best_id = 0, -1

	with open(fname, "r") as f:
		csv_reader = csv.reader(f, delimiter=",")
		for row in csv_reader: # each row is a list of str
			rows.append(row)

	# given the filename, check what was parsed
	# example fname: logs_ep_DATE.csv
	name_split = fname.split("_")
	ep, e, p = True, False, False
	relevant_metrics = {"test_events_mi_F1", "test_events_ma_F1", "test_participants_mi_F1", "test_participants_ma_F1"}
	if name_split[1] == "e":
		ep = False
		e = True
		relevant_metrics = {"test_events_mi_F1", "test_events_ma_F1"}
	elif name_split[1] == "p":
		ep = False
		p = True
		relevant_metrics = {"test_participants_mi_F1", "test_participants_ma_F1"}
	
	if name_split[1] not in {"ep", "p", "e"}:
		sys.exit("The csv file has an unexpected name, please rename it or modify this code")

	# iterate over runs to find the best one
	indices = dict()
	#import pdb; pdb.set_trace()
	for j, row in enumerate(rows):
		if j == 0:
			# get indices of relevant metrics names
			for rm in relevant_metrics:
				indices[rm] = row.index(rm)
		# go over runs
		if j > 0:
			current_avg = 0
			for metric, idx in indices.items():
				current_avg += float(row[idx])
			current_avg /= len(indices)
			print("index %s, average %f" % (row[0], current_avg))
			if current_avg > best_avg:
				best_avg = current_avg
				best_id = j # NOTE: this is not the experiment ID, but rather the csv row ID


	print("\t Best run info")
	for (d, val) in zip(rows[0], rows[best_id]):
		print(d, "\t", val)

# run
compare_runs(args.file)	
