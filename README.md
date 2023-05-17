# GraphAlgotithems
## classes:
### Node
#### elesments:
* missionaries-The number of missionaries in left side.
* cannibals-The number of cannibals in left side.
* side-the Side of boat in that situation on the left side.
*	isVisited- Say if we visit this node in the algorithm.
*	neightbors- The sons of that node. 
They will be just the nodes who legal to get them 
That is to say neighbor will be one in which the number of missionaries does not decrease from the number of cannibals and also his father will observe this rule, and that the transition between his father and him is the result of adding / subtracting respectively to the side (for 1 minus 0 plus) of the legal vectors (created from the restriction on the boat that can 1 or 2 but at least 1) which are <1,0,1,>, <0,1,1,>,<1,1,1>,<2,0,1>,<0,2,1>.
*	father- His father.
* h- used in GFS as heuristics, and in A* as f(n)=heuristics+result fromshort path from source to dest
#### fanction:
*	public void addNeighbor(Node n) -Add son to the node
*	public int compareTo(Node n) - help for pQue in graph to order the node in the queue
### Graph
#### elesments:
* nodes - Array that cut in nodes[m][c][s] the Node object that his number of missionaries=m,and his number of cannibals =c,and the boat is in side s.
* vectors - Vectors who help us find the neighbors of each node in    nodes.
* pQueue - Help us use priority pueue who sort by node.h in algorithms: DIJKSTRA, GBFS, A* 
* maxNumOfEdge- The nux num of age in case full graph
#### constructors:
1.	public Graph() - Creates the node without h element and update for each node his neighbors.(we call him in BFS and IDDFS)
2.	public Graph(Node source, Node dest) - use in GBFS
    Because GBFS wants find the shortest path in number of edges, we can assume every edge in capacity 1.
	  WE do DIJKSTRA for finding the sortest path frome source to dest and update
	  it on node. Because this is undirected graph and every edges have capacity of 1.
	  after we do DIJKSTRA we can get the shortest path of some node to dest.
	  by do Math.abs(dest.getH() - n.getH()) we get the shortest path from n to
	  dest.
    The constructor call to 1 and for each node n update his h to the shortest path from n to dest
    we call it in GBFS therefore his heuristic H1 is:
    for each n in nodes:
    H1=The shortest path from n to dest.
3. 	public Graph(Node source, Node dest, boolean isA) -use in A*  
   	Call to 1 and for each node n update his h to f(n)=shortest path from source to dest+H2
    we call it in GBFS therefore his heuristic H2 is:
    arrayOfH={ 18, 16, 14, 12, 10, 8, 6, 4, 2,0 }
    for each n in nodes:
    i is the shortest path from source to dest in n
    if  i > high of tree from source to dest of DIJKSTRA
	  H2=i-high
	  else
	  H2=arrayOfH[i].
#### fanction:
* public void theAddNeighbors(Node n)- get Node n and update him after it calculate n + vectors[i] for all i and checks the neighbor legal by all the rules (in 
Node explenation)
* public void theDiffNeighbors(Node n) - get Node n and update him after it calculate n - vectors[i] for all i and checks the neighbor legal by all the rules (in 
Node explenation)
* public void DIJKSTRA(Node source, Node dest) -  
for all n in nodes
n.h=the shortpath frome source to dest.
*	public void clearVisitedNodes()- clear visited vertecs to the next algorithem
*	public void getPath(ArrayList<Node> path, Node neightbor) - update path to the path from root(source) to neighbor
### GraphAlgorithms- extends Graph
#### elesments:
  no element
#### constructors:
3 constructor who call the 3 constructor in Graph respectively.
#### fanction:
*	public int BFS(ArrayList<Node> path, Node n, Node dest)- find shortest path by edges in a way of pass by degree
* private int DLS(ArrayList<Node> path, Node source, Node dest, int limit, int counter)-do DSL (like DFS  but from the last) from source to dest but if it doesent find it eliminate by limit
*	public int IDDFS(ArrayList<Node> path, Node n, Node dest)- find shortest path by edges in a way of pass by deap high times
*	public int GBFS(ArrayList<Node> path, Node n, Node dest)- find shortest path by edges in a way of pass by degree and sort the huiristic 
*	private int A(ArrayList<Node> path, Node n, Node dest)-  find shortest path by edges in a way of sort the f(n)=shortest path from source to dest+H2
* public static void run() - run the three algorithms on case source=<3,1,1>, dest= <0,0,0>.
### Main
  main function call run.
## Huiristic: 
### H1 is consistent beacause for every edge e we assume Capacity(e)=1 .
  and H1 is DIJKSTRA from n to dest 
  so for all edge e (source,dest)
  1=|source.h-dest.h|
  end if H1= infinity we will change to maxNumOfEdge that we know the other will led to it.
  therefore GBFS will give us optimal path.
### H2 is consistent beacause for every edge e we assume Capacity(e)=2 .
####  <3,3,1>=18
#### <3,2,0>,<2,2,0><3,1,0>=16
#### <3,2,1>=14
#### <2,1,0>,<3,0,0>=12
#### <3,1,1>,<2,2,1>=10
#### <2,0,0>,<1,1,0>=8
#### <3,0,1>,<2,1,1>=6
#### <1,0,0>=4
#### <2,0,1>,<1,1,1>=2
#### <0,0,0>=0
#### <1,0,1>=1
#### other:992
 beacause the edge is just between node who its h is not 992
 and for all line their neighbor can be or from one line befor or from one line after and the edge has 2 capaity this is consistance
 ## All the algorithm give a optimalic path.
 ## How to run
###Go to main file and run.
![image](https://github.com/talStamker/GraphAlgotithems/assets/89009470/0f1df3b3-76a3-4387-bedb-2d63fe4d2073)

###You will get it:
![image](https://github.com/talStamker/GraphAlgotithems/assets/89009470/1c4a390a-c895-4ac4-888f-3e3e5d01cb6c)
![image](https://github.com/talStamker/GraphAlgotithems/assets/89009470/f156b088-9309-403c-808b-16d76332bcd8)
	
