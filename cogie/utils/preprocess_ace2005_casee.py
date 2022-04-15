import os
import json
import collections
import copy

def process_ace2005_for_casee(raw_data_path,processed_data_path):
    """
    The main processing is to divide multiple events in a sentence into different samples and create schema.
    ##########################original data sample of ace2005 are shown below#################################
    {
    "sentence": "Even as the secretary of homeland security was putting his people on high alert last month,
    a 30-foot Cuban patrol boat with four heavily armed men landed on American shores,
     utterly undetected by the Coast Guard Secretary Ridge now leads.",
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
    "words": ["Even","as","the","secretary","of","homeland","security","was","putting",
    "his","people","on","high","alert","last","month",",","a","30-foot","Cuban","patrol",
    "boat","with","four","heavily","armed","men","landed","on","American","shores",",",
    "utterly","undetected","by","the","Coast","Guard","Secretary","Ridge","now","leads", "." ],

    }
    ##########################processed data sample of ace2005 are shown below################################
    train.json/dev.json/test.json
    {
    "id":"0",
    "content":  ["Even","as","the","secretary","of","homeland","security","was","putting",
    "his","people","on","high","alert","last","month",",","a","30-foot","Cuban","patrol",
    "boat","with","four","heavily","armed","men","landed","on","American","shores",",",
    "utterly","undetected","by","the","Coast","Guard","Secretary","Ridge","now","leads", "." ],
    "occur": ["Movement:Transport"],      #When empty  []
    "type":  "Movement:Transport",        #When empty  "<unk>"
    "triggers": [[27, 28]],               #When empty  []
    "index": 0,                           #When empty  None
    "args": {"Vehicle": [[17, 27]], "Artifact": [[23, 27]], "Destination": [[29, 31]]}     #空的时候为{}
    }
    schema.json
    {“event_type:[role_1,role_2,role_3]”，...}
    ############################################################################################################
    """
    raw_data_file_list=[{"raw_path":os.path.join(raw_data_path,data_type),
                         "processed_path":os.path.join(processed_data_path,data_type),
                         "old_add_id_path":os.path.join(processed_data_path,"old_add_id_"+data_type)}
                        for data_type in ["train.json","dev.json","test.json"]]
    schema = collections.defaultdict(set)
    id=-1
    for path_dict in raw_data_file_list:
        raw_path=path_dict["raw_path"]
        processed_path=path_dict["processed_path"]
        old_add_id_path=path_dict["old_add_id_path"]
        new_data=[]
        old_add_id_data=[]
        add_id_event=[]
        add_id_event_dict={}
        with open (raw_path) as f:
            new_line_dict={}
            old_add_id_data_dict={}
            raw_data=json.load(f)
            for step,line_dict in enumerate(raw_data):
                id=id+1
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
                        add_id_event_dict["type"]=event["event_type"]
                        args_value=collections.defaultdict(list)
                        add_id_event_dict["args"]={}
                        for role_item in event["arguments"]:
                            args_value[role_item["role"]].append([role_item["start"],role_item["end"]])
                            schema[event["event_type"]].add(role_item["role"])
                            add_id_event_dict["args"][role_item["role"]]=[]
                            add_id_event_dict["args"][role_item["role"]].append({"span":[role_item["start"],role_item["end"]],"word":role_item["text"]})
                        new_line_dict["args"]=args_value
                        new_line_dict["occur"]=occur
                        new_line_dict["triggers"]=triggers
                        add_id_event_dict["triggers"]={"span":[event["trigger"]["start"],event["trigger"]["end"]],"word":event["trigger"]["text"]}
                        new_line_dict["id"] = str(id)
                        new_data.append(new_line_dict)
                        new_line_dict={}
                        add_id_event.append(add_id_event_dict)
                        add_id_event_dict = {}
                    old_add_id_data.append({"id": str(id), "events":add_id_event})
                    add_id_event = []
                if len(line_dict["golden-event-mentions"])==0:
                    new_line_dict["content"] = line_dict["words"]
                    new_line_dict["index"] = None
                    new_line_dict["type"] ="<unk>"
                    new_line_dict["args"] = {}
                    new_line_dict["occur"] = []
                    new_line_dict["triggers"] = []
                    new_line_dict["id"]=str(id)
                    old_add_id_data.append({"id":str(id),"events":[]})
                    new_data.append(new_line_dict)
                    new_line_dict = {}
        with open(processed_path, 'w', encoding='utf-8') as f:
            for line in new_data:
                line = json.dumps(line, ensure_ascii=False)
                f.write(line + '\n')
        with open(old_add_id_path, 'w', encoding='utf-8') as f:
            for line in old_add_id_data:
                line = json.dumps(line, ensure_ascii=False)
                f.write(line + '\n')

    schema_list_dict={}
    for key,value in schema.items():
        schema_list_dict[key]=list(value)
    json.dump(schema_list_dict, open(os.path.join(processed_data_path,"schema.json"), "w"), indent=4)


if __name__=="__main__":
    process_ace2005_for_casee(raw_data_path="../../../cognlp/data/ee/ace2005/data",
                              processed_data_path="../../../cognlp/data/ee/ace2005casee/data")
