### Data Preparation

Go to bliss directory and compile 
```Shell
cd bliss/
make
```

Go to graph_converter directory and compile 
```Shell
cd graph_converter/
make
```

In graph_converter directory, run prepare_data.sh to download graph data and transfer the data format to our system's format. 
The shell script prepare_data.sh will also labelize the downloaded graphs
```Shell
bash prepare_data.sh  
```

Now you can see some directories in data/bin_graph/

### Compile
Go to home directory and compile the project
```Shell
make clean
make
```

## Run
```Shell
#Unlabeled 
./bin/ulb.out ./data/bin_graph/com-youtube.ungraph/snap.txt data/pattern/1.g
#Labeled 
./bin/lb.out ./data/bin_graph/com-youtube.ungraph/snap.txt data/pattern/1.g
```

### Adding Vertex Labels
Vertex labels are kept in "snap.txt.label.bin". To adjust number of labels in the graph, change 4 to number of labels you'd like to test at https://github.com/lyuheng/tdfs/blob/main/graph_converter/labelize.cpp#L50.
```
labels[i]=std::rand()%4;
```

Then under folder graph_converter, type make to compile. Finally rerun "labelize".
```
cd graph_converter
make
./bin/labelize ./data/bin_graph/com-youtube.ungraph/snap.txt
```


