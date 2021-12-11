"""
Query data in mongodb.
liyuejiang
2021.11.23
"""
import os
import argparse
import json
import time
import pymongo
from tqdm import tqdm


FIELDS = ["Time", "Verify", "Area",
    "Sex", "City", "OriAuthor", "OriTxt"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--author", "-a", type=str, required=True,
        help="Original author to be queried.")
    parser.add_argument("--dump_path", "-p", type=str, default="./",
        help="Path to save dumped results.")
    parser.add_argument("--dump_name", "-n", type=str, default="query_res",
        help="Name of the query result files.")
    parser.add_argument("--fields", "-f", type=str, nargs="+", default=["Time", "OriAuthor", "OriTxt"],
        help="Fields to be saved. OriTxt must be included.")
    parser.add_argument("--encoding", type=str, default="gb18030")
    
    args = parser.parse_args()
    field_sets = set(args.fields)
    if not field_sets.issubset(set(FIELDS)):
        raise ValueError("Field selection error.")
    if "OriTxt" not in field_sets:
        raise ValueError("OriTxt field must be in fields argumet")
    
    return args


def proc_query(collection, query, args):
    """
    processing the data query, and export data into disk.
    Currently, the query is based on original users.
    It group the results based on the original contents,
    and return list of original weibo (including its repost information).
    """

    query_fields = {f: 1 for f in args.fields}
    query_fields["_id"] = 0
    content_map = {}
    content_id = 0
    repost_file = os.path.join(args.dump_path, f"{args.dump_name}_repost.txt")
    ori_file = os.path.join(args.dump_path, f"{args.dump_name}_orimap.txt")

    print("Quering and dumping data...")
    start_time = time.time()
    res = collection.find(query, query_fields)
    repo_num = 0
    with open(repost_file, "w", encoding=args.encoding) as f_rec:
        f_rec.write(''.join(args.fields))

        for doc in tqdm(res):
            if doc["OriTxt"] in content_map:
                doc["OriTxt"] = content_map[doc["OriTxt"]]
            else:
                content_map[doc["OriTxt"]] = content_id
                doc["OriTxt"] = content_id
                content_id += 1
            
            row = [str(doc[f]) for f in args.fields]
            cur_row = ",".join(row) + "\n"
            f_rec.write(cur_row)
            repo_num += 1

    end_time = time.time()
    print(f"Find {content_id} original contents by {args.author}, {repo_num} reposts in total.")
    print(f"Finish querying and dumping. Time cosumes {(end_time - start_time):.2f} s.")
    print()

    print("Dumping original content mapping...")
    start_time = time.time()
    with open(ori_file, "w", encoding=args.encoding) as f:
        json.dump(content_map, f)
    end_time = time.time()
    print(f"Finish dumping original content mapping. \
Time cosumes {(end_time - start_time):.2f} s.")
    print()


if __name__ == "__main__":
    client = pymongo.MongoClient()
    db = client['YiqingWeibo']
    collection = db["reposts"]
    args = get_args()
    for arg in vars(args):
        print(f"Argument {arg} = {getattr(args, arg)}")

    stage_query = {
        "OriAuthor": args.author
    }    
    
    proc_query(db.reposts, stage_query, args)
    print("Query finished.")