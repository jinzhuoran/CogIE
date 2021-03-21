"""
@Author: jinzhuan
@File: cognet.py
@Desc: 
"""
COGNET_ENDPOINT = "http://159.226.21.226/cognet_sparql/repositories/cognet"  # 查询接口地址，把这个放到配置文件里

import requests

accept_mapping = {
    "json": "application/sparql-results+json",
    "rdf+json": "application/rdf+json",
    "json-ld": "application/ld+json"
}


class JSON_SPARQLEndpoint(object):
    def __init__(self, endpoint, return_format="json", method="POST"):
        self.endpoint = endpoint
        self.return_format = return_format
        self.method = method
        self.headers = {"Accept": accept_mapping[self.return_format],
                        "Content-Type": "application/x-www-form-urlencoded"}

    def query(self, query_string, timeout=10, limit=1000, offset=None):
        data = {
            "query": query_string,
            "timeout": timeout
        }
        if limit:
            data['limit'] = limit
        if offset:
            data['offset'] = offset
        result = requests.request(self.method, self.endpoint, data=data, headers=self.headers, timeout=timeout)
        if result.ok:
            return result.json()
        else:
            raise Exception(result.text)


class CognetServer:
    def __init__(self, endpoint="http://159.226.21.226/cognet_sparql/repositories/cognet"):
        # 连接查询接口
        self.endpoint = JSON_SPARQLEndpoint(endpoint)

    # "<http://www.wikidata.org/entity/Q16251131>"
    def query(self, sql):
        result = self.endpoint.query("""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        select ?e where { 
            values ?wd {$wikidata} 
        	?e owl:sameAs ?wd.
        } 
        """.replace("$wikidata", sql))
        if len(result['results']['bindings']) == 0:
            return "unk"
        else:
            return result['results']['bindings'][0]['e']['value']
