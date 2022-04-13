import os
import json
import collections

def write(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')

def process_ace2005_for_casee(raw_data_path,processed_data_path):
    raw_data_file_list=[{"raw_path":os.path.join(raw_data_path,data_type),
                         "processed_path":os.path.join(processed_data_path,data_type)}
                        for data_type in ["train.json","dev.json","test.json"]]
    schema = collections.defaultdict(set)
    for path_dict in raw_data_file_list:
        raw_path=path_dict["raw_path"]
        processed_path=path_dict["processed_path"]
        new_data=[]
        with open (raw_path) as f:
            new_line_dict={}
            raw_data=json.load(f)
            for line_dict in raw_data:
                occur=[]
                triggers=[]
                if len(line_dict["golden-event-mentions"])>0:
                    for event in line_dict["golden-event-mentions"]:
                        occur.append(event["event_type"])
                        triggers.append([event["trigger"]["start"],event["trigger"]["end"]])
                    for index,event in enumerate(line_dict["golden-event-mentions"]):
                        new_line_dict["content"] = line_dict["words"]
                        new_line_dict["index"]=index
                        new_line_dict["type"]=event["event_type"]
                        args_value=collections.defaultdict(list)
                        for role_item in event["arguments"]:
                            args_value[role_item["role"]].append([role_item["start"],role_item["end"]])
                            schema[event["event_type"]].add(role_item["role"])
                        new_line_dict["args"]=args_value
                        new_line_dict["occur"]=occur
                        new_line_dict["triggers"]=triggers
                        new_data.append(new_line_dict)
                        new_line_dict={}
                if len(line_dict["golden-event-mentions"])==0:
                    new_line_dict["content"] = line_dict["words"]
                    new_line_dict["index"] = -1
                    new_line_dict["type"] ="<unk>"
                    new_line_dict["args"] = {}
                    new_line_dict["occur"] = []
                    new_line_dict["triggers"] = []
                    new_data.append(new_line_dict)
                    new_line_dict = {}
        write(new_data, processed_path)
    schema_list_dict={}
    for key,value in schema.items():
        schema_list_dict[key]=list(value)
    json.dump(schema_list_dict, open(os.path.join(processed_data_path,"schema.json"), "w"), indent=4)


if __name__=="__main__":
    process_ace2005_for_casee(raw_data_path="../../../cognlp/data/ee/ace2005/data",
                              processed_data_path="../../../cognlp/data/ee/ace2005casee/data")

"""
##########################ace2005的部分原始数据如下所示##########################
{
"sentence": "Even as the secretary of homeland security was putting his people on high alert last month, a 30-foot Cuban patrol boat with four heavily armed men landed on American shores, utterly undetected by the Coast Guard Secretary Ridge now leads.",
"golden-event-mentions": [
  {
    "trigger": {
      "text": "landed",
      "start": 27,
      "end": 28
    },
    "arguments": [
      {
        "role": "Vehicle",
        "entity-type": "VEH:Water",
        "text": "a 30-foot Cuban patrol boat with four heavily armed men",
        "start": 17,
        "end": 27
      },
      {
        "role": "Artifact",
        "entity-type": "PER:Group",
        "text": "four heavily armed men",
        "start": 23,
        "end": 27
      },
      {
        "role": "Destination",
        "entity-type": "LOC:Region-General",
        "text": "American shores",
        "start": 29,
        "end": 31
      }
    ],
    "event_type": "Movement:Transport"
  }
],
"words": ["Even","as","the","secretary","of","homeland","security","was","putting","his","people","on","high","alert","last","month",",","a","30-foot",
          "Cuban","patrol","boat","with","four","heavily","armed","men","landed","on","American","shores",",","utterly","undetected","by","the","Coast","Guard","Secretary","Ridge","now","leads", "." ],

}
##########################根据casee，预处理为如下格式，主要处理为把一个句子里的多个事件，分为不同的sample##########################
{
"id": "58880e1cb716866992ac65cb3f2f2c03", 
"content":  ["Even","as","the","secretary","of","homeland","security","was","putting","his","people","on","high","alert","last","month",",","a","30-foot",
          "Cuban","patrol","boat","with","four","heavily","armed","men","landed","on","American","shores",",","utterly","undetected","by","the","Coast","Guard","Secretary","Ridge","now","leads", "." ],
"occur": ["Movement:Transport"], 
"type":  "Movement:Transport", 
"triggers": [[27, 28]], 
"index": 0,
"args": {"Vehicle": [[17, 27]], "Artifact": [[23, 27]], "Destination": [[29, 31]]}
}
以及建立schema
{“event_type:[role_1,role_2,role_3]”，...}
"""