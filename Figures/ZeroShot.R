  library(ggplot2)
  library(dplyr)
  library(reshape2)
  library(stringr)
  library(tidyr)

  f<-list.files("/Users/benweinstein/Documents/BirdDetector/Figures/",pattern="result",full.names = T)
  f<-f[str_detect(f,"result_")]
  df<-bind_rows(lapply(f,read.csv))

  #Label datasets
  naming<-unique(df$test_set)
  nameframe <-data.frame(test_set=naming,newname=c("6. Albatross - Falklands","7. Marshbirds - Canada","13. Lake Michigan - USA","10. Seabirds - Indian Ocean","11. Pelicans - Utah","12. Ducks - New Mexico","2. Seabirds - South Pacific","3. Penguins - Antartica","5. Penguins and Shags - Antartica","9. Seabirds - North Atlantic","4. Terns - Guinea","8. Ducks - Cape Cod"))
  df<-df %>% inner_join(nameframe) %>% select(-test_set) %>% select(-X)
  df<-melt(df,id.vars=c("Model","newname","Annotations"))
  df[df$Model == "Zero Shot","Model"]<-"None"
  df[df$Model == "Fine Tune","Model"]<-"All available"
  df[df$Model == "Min Annotation","Model"]<-"1000 birds"

  min_annotation<-df %>% filter(Model %in% c("1000 birds")) %>% group_by(Model, newname, variable) %>%
    summarize(min=min(value),max=max(value), mean=mean(value))
  avgm<- df %>% filter(!Model %in% c("1000 birds"))
  # add min annotation for legend purposes
  m<-min_annotation %>% select(Model,newname, variable, value=mean) %>% bind_rows(.,avgm)

  #order of dataasets
  #ord<-m %>% group_by(newname) %>% filter(variable=="Recall") %>% summarize(s = mean(value)) %>% arrange(s) %>% .$newname
  m$newname <- factor(m$newname,levels = rev(str_sort(unique(m$newname),numeric = T)))
  ggplot(m) + geom_point(aes(x=newname,y=value,col=Model),size=2.5) +
    geom_errorbar(data=min_annotation %>% filter(Model=="1000 birds"), col="black",alpha=.6,width=0.1, size=1.5, aes(x=newname, ymin=min,ymax=max)) +
   coord_flip()+facet_wrap(~variable) + labs(x="Dataset",col="Local Annotations")  + scale_color_brewer(type="qual",palette = 2) + theme_bw()
  ggsave("Zeroshot.png",height=4,width=7)

  m %>% group_by(Model, variable) %>% summarize(mean = mean(value))
  m %>% group_by(newname,Model, variable) %>% summarize(mean = mean(value)) %>% filter(Model=="None") %>% as.data.frame()

  ggplot(m %>% filter(Model %in% c("None","1000 birds")),aes(x=value,fill=Model)) + geom_density(alpha=0.5) + facet_wrap(~variable)

  zenodo<-df %>% filter(Model == "Fine Tune")
  write.csv(zenodo,"finetuned_results.csv")



  #With and without pretraining
  global_weights<-df %>% filter(Model %in% c("1000 birds","RandomWeight"))
  global_weights[global_weights$Model == "1000 birds","Pretrained"] = TRUE
  global_weights[!global_weights$Model == "1000 birds","Pretrained"] = FALSE

  ggplot(global_weights,aes(x=newname,y=value, col=Pretrained)) + geom_boxplot() + facet_wrap(~variable) + coord_flip() + theme_bw()
  ggsave("Globalweights.png",height=5,width=8)

  ## Vector arrows
  df$newname<-factor(df$newname,levels = str_sort(as.character(unique(df$newname)),numeric = T))
  vector_data <- df %>%filter(Model %in% c("1000 birds","None")) %>% group_by(Model, newname, variable) %>% summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")
  ggplot(vector_data) + geom_point(aes(col=newname, x=`None_Recall`,y=`None_Precision`)) + geom_segment(show.legend =FALSE,aes(col=newname, x=`None_Recall`,y=`None_Precision`,xend=`1000 birds_Recall`,yend=`1000 birds_Precision`), arrow = arrow(length = unit(0.5, "cm"))) +
    labs(x="Recall",y="Precision", col="Dataset") + theme_bw()
  ggsave("Vector_arrows.png",height=4,width=6)
  #Boxplot with insets
  df %>% filter(Model == "None") %>% ggplot(.,aes(x=variable, y=value)) + geom_violin() + geom_point() + labs(x="") + ylim(0,1)
  ggsave("Boxplots_zeroshot.svg",height=4,width=4)

  vector_data <- df %>%filter(Model %in% c("1000 birds","None")) %>% group_by(Model, newname, variable) %>% summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")
  ggplot(vector_data) + geom_point(aes(col=newname, x=`None_Recall`,y=`None_Precision`)) + geom_segment(show.legend =FALSE,aes(col=newname, x=`None_Recall`,y=`None_Precision`,xend=`1000 birds_Recall`,yend=`1000 birds_Precision`), arrow = arrow(length = unit(0.5, "cm"))) +
    labs(x="Recall",y="Precision", col="Dataset") + theme_bw()
  ggsave("Vector_arrows.png",height=4,width=6)

  df %>% filter(Model == "None") %>% ggplot(.,aes(x=variable, y=value)) + geom_violin() + geom_point(aes(col=newname)) + labs(x="", col="Dataset") + ylim(0,1) +
    theme(legend.position = "bottom") + guides(col=guide_legend(ncol=2))

  ggsave("Violinplots.svg",height=7.5,width=5.5) + theme_bw()
  # zeo shot difference and random weights
  random_df <- df %>%
    filter(Model %in% c("RandomWeight")) %>% group_by(Model, Annotations,newname, variable) %>%
    summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")

  None_df <- df %>%
    filter(Model %in% c("1000 birds")) %>% group_by(Model, Annotations,newname, variable, Iteration) %>%
    summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")

  annotation_df <- random_df %>% inner_join(None_df) %>%
    mutate(Recall_diff = RandomWeight_Recall - `1000 birds_Recall`,Precision_diff = RandomWeight_Precision - `1000 birds_Precision`) %>%
    filter(!is.na(Annotations)) %>% select(newname, Annotations, Recall=Recall_diff, Precision=Precision_diff) %>%
    melt(id.vars=c("newname","Annotations"))

  ggplot(annotation_df,aes(x=Annotations,y=value,col=newname)) +
    facet_wrap(~variable, ncol=1) + geom_line() + geom_point(size=3) +
    geom_hline(yintercept = 0, linetype="dashed") +
    labs(y="Difference from Fine-tuned Model", x="Local Annotations",col="Dataset") + theme_bw() +
    scale_y_continuous(labels=scales::percent_format()) + xlim(0,10000)
  ggsave("Fine_tuned_Annotations.png",height=5,width=7.5)

  # zeo shot difference and random weights
  None_df <- df %>%
    filter(Model %in% c("None")) %>% group_by(Model,newname, variable) %>%
    summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")

  annotation_df <- random_df %>% inner_join(None_df) %>%
    mutate(Recall_diff = RandomWeight_Recall - None_Recall,Precision_diff = RandomWeight_Precision - None_Precision) %>%
    filter(!is.na(Annotations)) %>% select(newname, Annotations, Recall=Recall_diff, Precision=Precision_diff) %>%
    melt(id.vars=c("newname","Annotations"))

  ggplot(annotation_df,aes(x=Annotations,y=value,col=newname)) +
    facet_wrap(~variable) + geom_line() + geom_point() +
    geom_hline(yintercept = 0, linetype="dashed") +
    labs(y="Difference from Fine-tuned Model", x="Local Annotations",col="Dataset")

  #Local only vectors

  vector_data <- df %>%filter(Model %in% c("RandomWeight","All available")) %>%
    group_by(Model, newname, variable) %>%
    summarize(value=mean(value)) %>%
    pivot_wider(names_from = c("Model","variable"),values_from="value")
  ggplot(vector_data) + geom_point(aes(col=newname, x=`RandomWeight_Recall`,y=`RandomWeight_Precision`)) + geom_segment(show.legend =FALSE,aes(col=newname, x=`RandomWeight_Recall`,y=`RandomWeight_Precision`,xend=`All available_Recall`,yend=`All available_Precision`), arrow = arrow(length = unit(0.5, "cm"))) +
    labs(x="Recall",y="Precision", col="Dataset") + theme_bw()
  ggsave("Vector_arrows_finetune.png",height=4,width=6)

  vector_data <- df %>%filter(Model %in% c("RandomWeight","All available")) %>%
    group_by(Model, newname, variable, Annotations) %>%
    summarize(value=mean(value)) %>%
    pivot_wider(names_from = c("Model","variable"),values_from="value")
  ggplot(vector_data) + geom_point(aes(size=Annotations,col=newname, x=`RandomWeight_Recall`,y=`RandomWeight_Precision`)) +
    geom_segment(show.legend =FALSE,aes(col=newname, x=`RandomWeight_Recall`,y=`RandomWeight_Precision`,xend=`All available_Recall`,yend=`All available_Precision`), arrow = arrow(length = unit(0.25, "cm"))) +
    labs(x="Recall",y="Precision", col="Dataset") + theme_bw()
  ggsave("Vector_arrows_finetune_size.png",height=4,width=6)

  #AS Differences
  vector_data <- df %>%filter(Model %in% c("RandomWeight","All available")) %>%
    group_by(Model, newname, variable, Annotations) %>%
    summarize(value=mean(value)) %>%
    pivot_wider(names_from = c("Model","variable"),values_from="value") %>% mutate(Recall= `All available_Recall` - `RandomWeight_Recall`) %>%
    mutate(Precision = `All available_Precision` - `RandomWeight_Precision`) %>% select(Annotations, newname, Recall, Precision) %>%
    melt(id.vars=c("newname","Annotations"))

  ggplot(vector_data,aes(x=Annotations, y= value, col=newname), size=5) +
    facet_wrap(~variable, ncol=1) + geom_point()  +
    scale_y_continuous(labels = scales::percent_format()) +
    theme_bw() +
    geom_hline(yintercept=0,linetype="dashed") + labs(col="Dataset",y="Difference from Local Only Model")
