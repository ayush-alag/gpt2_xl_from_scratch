# DNS server: given a domain name, convert to IP address, IPs talk to each other via the IP layer
# designing a DNS cache
# internet provider has a DNS server (belongs to provider)
# sending a request to google -> get the IP from the DNS server -> send request to that IP
# TTL for each domain
import time
from collections import defaultdict, OrderedDict

class Client:
    def __init__(self, dns_cache):
        self.dns_cache = dns_cache
    
    # this is a function to actually make the network request
    # also returns a ttl
    def get_ip_dns_server(self, domain):
        DUMMY_IP = "1.1.2" # pretend this is from the server
        DUMMY_TTL = 30
        return DUMMY_IP, DUMMY_TTL
    
    def get_ip_addr(self, domain):
        maybe_cached_ip = self.dns_cache.get_ip(domain)
        if maybe_cached_ip:
            return maybe_cached_ip
        
        ip, ttl = self.get_ip_dns_server(domain)
        self.dns_cache.update(domain, ip, ttl)
        
class Node:
    def __init__(self, domain, ip, ending_time):
        self.domain = domain
        self.ip = ip
        self.ending_time = ending_time

class DNSCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.domain_to_node = {}
        self.ending_time_to_domains = defaultdict(list)
        self.node_list = []
    
    # return the ip address for this domain
    # if ttl is expired, we want to hit the DNS Server
    def get_ip(self, domain):
        if domain not in self.domain_to_node:
            return None
    
        node = self.domain_to_node[domain]
        if node.ending_time < time.time():
            return None
        
        self._move_node_to_front(domain, node)
        
        return node.ip
    
    def evict_cache(self):
        ordered_dict = OrderedDict(sorted(self.ending_time_to_domains.items()))
        min_ending_time = next(iter(ordered_dict))
        if min_ending_time < time.time():
            domain = self.ending_time_to_domains[min_ending_time][0]
            remove_node = self.domain_to_node[domain]
            domain = remove_node.domain
            ending_time = remove_node.ending_time
        else:
            remove_node = self.node_list[-1]
            domain = remove_node.domain
            ending_time = remove_node.ending_time
            
        self.node_list.remove(remove_node)
        del self.domain_to_node[domain]
        
        ending_time_domains = self.ending_time_to_domains[ending_time]
        ending_time_domains.remove(domain)
        if len(ending_time_domains) == 0:
            del self.ending_time_to_domains[ending_time]
    
    def _move_node_to_front(self, domain, node):
        self.node_list.remove(node)
        self.node_list.insert(0, node)
        self.domain_to_node[domain] = node 
    
    def update(self, domain, ip, ttl):
        if len(self.domain_to_node) == self.max_size:
            self.evict_cache()
            
        ending_time = time.time() + ttl
        
        if domain in self.domain_to_node:
            node = self.domain_to_node[domain]
            old_ending_time = node.ending_time
            self.ending_time_to_domains[old_ending_time].remove(node.domain)
            node.ending_time = ending_time
            self._move_node_to_front(domain, node)
        else:
            node = Node(domain, ip, ending_time)
            self.node_list.append(node)
            self._move_node_to_front(domain, node)

        self.ending_time_to_domains[ending_time].append(domain)

# map of (ending_time -> list of domains) : ending_time_to_domains
# cache of domain -> Node : domain_to_nodes
# linked list of Nodes (domain, ending_time) : node_list

# case 1: check the first ending_time in ending_time_to_domains (sorted), 
# and if its expired, we get a domain from the value
# and then lookup into domain_to_nodes to get the Node object, remove it from the node_list,
# and remove from  domain_to_nodes

# case 2: no expired times, so we get from the end of node_list (domain, ending time)
# delete the relevant entry in domain_to_nodes
# and also delete the relevant entry in ending_time_to_domains