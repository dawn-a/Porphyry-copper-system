library(stringr)
library(readxl)
library(xlsx)
library(openxlsx)
library(arules)
library(arulesViz)
library(dplyr)
library(data.table)
library(igraph)

Porphyry_deposit_all <- read.xlsx("./Porphyry_deposit.xlsx")

df <- subset(Porphyry_deposit_all,!Porphyry_deposit_all$Sub_type == "Cu-Mo-Au")

data<-Porphyry_deposit_all[which(str_count(Porphyry_deposit_all$Mineral_assemblage,",")>3),]

Type_count=count(data_1,Sub_type)

mineral_type_merge <- paste(data_1$Sub_type,data_1$Mineral_assemblage,sep = ',')

associated.mins <- strsplit(as.character(mineral_type_merge),',')

type=c("Cu-Mo","Cu-Au","Cu")

all_rules <- apriori(associated.mins, 
                     parameter = list(supp = 0.04, conf = 0.7,target = "rules",maxtime = 0,minlen = 2,maxlen=30),
                     appearance = list(rhs = type, default = 'lhs'))


rules.sub.Cu.Au <- subset(all_rules, subset = (rhs %in% "Cu-Au"))
rules.sub.Cu.Mo <- subset(all_rules, subset = (rhs %in% "Cu-Mo"))
rules.sub.Cu <- subset(all_rules, subset = (rhs %in% "Cu"))


inspect(head(rules.sub.Cu,n=10))
inspect(head(rules.sub.Cu.Au,n=10))
inspect(head(rules.sub.Cu.Mo,n=10))

rules.sub.Mo.list<-as(rules.sub.Cu.Mo@lhs,"list")
rules.sub.Au.list<-as(rules.sub.Cu.Au@lhs,"list")
rules.sub.Cu.list<-as(rules.sub.Cu@lhs,"list")

items.Mo<-data.frame(matrix(ncol = 2, nrow = sum(lengths(rules.sub.Mo.list))))
items.Au<-data.frame(matrix(ncol = 2, nrow = sum(lengths(rules.sub.Au.list))))
items.Cu<-data.frame(matrix(ncol = 2, nrow = sum(lengths(rules.sub.Cu.list))))

colnames(items.Mo) <- c('id','mineral')
colnames(items.Au) <- c('id','mineral')
colnames(items.Cu) <- c('id','mineral')

items.Cu <- rbindlist(
  lapply(seq_along(rules.sub.Cu.list), function(i) {
    if(length(rules.sub.Cu.list[[i]]) > 0) {
      data.table(id = i, mineral = rules.sub.Cu.list[[i]])
    }
  })
)
items.Au <- rbindlist(
  lapply(seq_along(rules.sub.Au.list), function(i) {
    if(length(rules.sub.Au.list[[i]]) > 0) {
      data.table(id = i, mineral = rules.sub.Au.list[[i]])
    }
  })
)
items.Mo <- rbindlist(
  lapply(seq_along(rules.sub.Mo.list), function(i) {
    if(length(rules.sub.Mo.list[[i]]) > 0) {
      data.table(id = i, mineral = rules.sub.Mo.list[[i]])
    }
  })
)


count.Mo=count(items.Mo,mineral)
count.Au=count(items.Au,mineral)
count.Cu=count(items.Cu,mineral)

mineral_links_Mo=data.frame(matrix(ncol = 2, nrow = 0))
colnames(mineral_links_Mo) <- c('source','target')
items_dt_Mo <- as.data.table(items.Mo)
mineral_links_Mo <- items_dt_Mo[, {
  if(.N >= 2) {
    combns <- combn(mineral, 2, simplify = FALSE)
    .(source = sapply(combns, `[`, 1), 
      target = sapply(combns, `[`, 2))
  }
}, by = id][, .N, by = .(source, target)]


mineral_links_Au=data.frame(matrix(ncol = 2, nrow = 0))
colnames(mineral_links_Au) <- c('source','target')
items_dt_Au <- as.data.table(items.Au)
mineral_links_Au <- items_dt_Au[, {
  if(.N >= 2) {
    combns <- combn(mineral, 2, simplify = FALSE)
    .(source = sapply(combns, `[`, 1), 
      target = sapply(combns, `[`, 2))
  }
}, by = id][, .N, by = .(source, target)]


mineral_links_Cu=data.frame(matrix(ncol = 2, nrow = 0))
colnames(mineral_links_Cu) <- c('source','target')
items_dt_Cu <- as.data.table(items.Cu)
mineral_links_Cu <- items_dt_Cu[, {
  if(.N >= 2) {
    combns <- combn(mineral, 2, simplify = FALSE)
    .(source = sapply(combns, `[`, 1), 
      target = sapply(combns, `[`, 2))
  }
}, by = id][, .N, by = .(source, target)]


write.xlsx(mineral_links_Mo,file = 'mineral_Links_Mo.xlsx')
write.xlsx(mineral_links_Au,file = 'mineral_Links_Au.xlsx')
write.xlsx(mineral_links_Cu,file = 'mineral_Links_Cu.xlsx')


mineral_nodes=read.xlsx("./Nodes.xlsx")

Mo_data=subset(mineral_nodes,mineral_nodes$target=='Cu-Mo')
Mo_nodes=data.frame(mineral=Mo_data$source,frequency=Mo_data$frequency)
Au_data=subset(mineral_nodes,mineral_nodes$target=='Cu-Au')
Au_nodes=data.frame(mineral=Au_data$source,frequency=Au_data$frequency)
Cu_data=subset(mineral_nodes,mineral_nodes$target=='Cu')
Cu_nodes=data.frame(mineral=Cu_data$source,frequency=Cu_data$frequency)

Mo_links=read.xlsx("./mineral_Links_Mo.xlsx")
Mo_edges=as.matrix(Mo_links[,1:2])

Au_links=read.xlsx("./mineral_Links_Au.xlsx")
Au_edges=as.matrix(Au_links[,1:2])

Cu_links=read.xlsx("./mineral_Links_Cu.xlsx")
Cu_edges=as.matrix(Cu_links[,1:2])



Mo_graph <- graph_from_data_frame(d = Mo_edges, vertices = Mo_nodes, directed = FALSE)
Mo_v_size <- Mo_nodes$frequency*30

Au_graph <- graph_from_data_frame(d = Au_edges, vertices = Au_nodes, directed = FALSE)
Au_v_size <- Au_nodes$frequency*30

Cu_graph <- graph_from_data_frame(d = Cu_edges, vertices = Cu_nodes, directed = FALSE)
Cu_v_size <- Cu_nodes$frequency*30


E(Mo_graph)$weight <- log(Mo_links$n)
E(Au_graph)$weight <- log(Au_links$n+10)
E(Cu_graph)$weight <- Cu_links$n





Mo_layout <- layout_with_fr(Mo_graph)
plot(Mo_graph, layout = Mo_layout, vertex.size = Mo_v_size, vertex.color = "steelblue", 
     edge.color = "gray",edge.width=log(Mo_links$n+1)/4)

Au_layout <- layout_with_fr(Au_graph)
plot(Au_graph, layout = Au_layout, vertex.size = Au_v_size, vertex.color = "steelblue", 
     edge.color = "gray",edge.width=log(Au_links$n+1)/4)

Cu_layout <- layout_with_fr(Cu_graph)
plot(Cu_graph, layout = Cu_layout, vertex.size = Cu_v_size, vertex.color = "steelblue", 
     edge.color = "gray",edge.width=log(Cu_links$n+1)/2)


All_nodes = data.frame(mineral=mineral_nodes$source,type=1)
All_nodes[86,1]="Cu"
All_nodes[87,1]="Cu-Au"
All_nodes[88,1]="Cu-Mo"
Bi_nodes = count(All_nodes,mineral)
Bi_nodes[16,2]=0
Bi_nodes[17,2]=0
Bi_nodes[18,2]=0
write.xlsx(Bi_nodes,file = 'Bi_nodes.xlsx')
Bi_nodes=read.xlsx("./Bi_nodes.xlsx")

type_colors <- c(
  "2" = "steelblue",   # 将 "Type_A" 映射为 steelblue
  "3" = "coral",       # 将 "Type_B" 映射为 coral
  "1" = "lightgreen",
  "0" = "red"# 将 "Type_C" 映射为 lightgreen
  # ... 根据您实际的 type 类型继续添加
)
vertex_colors <- type_colors[as.character(Bi_nodes$n)]

All_links = data.frame(source=mineral_nodes$source,target=mineral_nodes$target,value=mineral_nodes$frequency)
All_edges=as.matrix(All_links[,1:2])

All_graph <- graph_from_data_frame(d = All_edges, vertices = Bi_nodes, directed = FALSE)
All_v_size <- Bi_nodes$value*10

E(All_graph)$weight <- 5
All_layout <- layout_with_fr(All_graph)

plot(All_graph, layout = All_layout, vertex.size = All_v_size, vertex.color = vertex_colors, 
     edge.color = "gray",edge.width=0.5)
