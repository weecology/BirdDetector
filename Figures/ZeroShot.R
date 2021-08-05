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
  df<-df %>% inner_join(nameframe) %>% select(-test_set) %>% select(-X, -Iteration)
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

  df %>% filter(Model %in% c("1000 birds", "All available")) %>% filter(Annotations == 1000 & Model == "1000 birds" | Model=="All available") %>% group_by(Model, newname, variable) %>% summarize(value = mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value") %>%
    ggplot(.,aes(x))  + geom_point(aes(col=newname, x=`1000 birds_Recall`,y=`1000 birds_Precision`)) + geom_segment(show.legend =FALSE,aes(col=newname, xend=`All available_Recall`,yend=`All available_Precision`,x=`1000 birds_Recall`,y=`1000 birds_Precision`), arrow = arrow(length = unit(0.5, "cm"))) +
    labs(x="Recall",y="Precision", col="Dataset") + theme_bw()
  ggsave("All_available_vectors.png",height=4,width=6)

  ggplot(m %>% filter(Model %in% c("None","1000 birds")),aes(x=value,fill=Model)) + geom_density(alpha=0.5) + facet_wrap(~variable)

  zenodo<-df %>% filter(Model == "Fine Tune")
  write.csv(zenodo,"finetuned_results.csv")

  #With and without pretraining
  global_weights<-df %>% filter(Model %in% c("1000 birds","RandomWeight"))
  global_weights[global_weights$Model == "1000 birds","Pretrained"] = TRUE
  global_weights[!global_weights$Model == "1000 birds","Pretrained"] = FALSE
  global_weights$newname <- factor(global_weights$newname, levels=rev(str_sort(unique(global_weights$newname), numeric = T)))
  ggplot(global_weights,aes(x=newname,y=value, col=Pretrained)) + geom_boxplot(fill="grey90", size=0.5) + facet_wrap(Annotations~variable) + coord_flip() + theme_bw()
  ggsave("Globalweights.png",height=6,width=8)

    #Boxplot with insets
  vardata <- df %>% filter(Model %in% c("1000 birds","RandomWeight")) %>% group_by(Model, Annotations, variable) %>% summarize(var=var(value))
  vardata[vardata$Model == "1000 birds","Pretrained"] = TRUE
  vardata[!vardata$Model == "1000 birds","Pretrained"] = FALSE

    ggplot(vardata,aes(x=Annotations, y=var, shape=variable, col=Pretrained)) + geom_line() + geom_point(col="black") +
    labs(x="Local Annotations",fill="Dataset", y="Variance") + labs(shape="Metric") + theme_bw()
  ggsave("Variance.png",height=4,width=6)

  df$newname<-factor(df$newname,levels = str_sort(as.character(unique(df$newname)),numeric = T))
  vector_data <- df %>%filter(Model %in% c("1000 birds","None")) %>% group_by(Model, newname, variable) %>% filter(Annotations==1000|is.na(Annotations)) %>% summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")
  ggplot(vector_data) + geom_point(aes(col=newname, x=`None_Recall`,y=`None_Precision`)) + geom_segment(show.legend =FALSE,aes(col=newname, x=`None_Recall`,y=`None_Precision`,xend=`1000 birds_Recall`,yend=`1000 birds_Precision`), arrow = arrow(length = unit(0.5, "cm"))) +
    labs(x="Recall",y="Precision", col="Dataset") + theme_bw()
  ggsave("Vector_arrows.png",height=4,width=6)
  ggsave("Vector_arrows.svg",height=4,width=6)

  violin_data<-df %>% filter(Model %in% c("None","1000 birds")) %>%
    filter(is.na(Annotations) | Annotations == 1000) %>%
    group_by(Model, newname, Annotations, variable) %>% summarize(value=mean(value)) %>% mutate(Model = as.character(Model))
  violin_data[violin_data$Model == "None","Model"] <- "0"
  violin_data$Model[violin_data$Model == "1000 birds"] <- "1000"
  violin_data$Model <- as.numeric(violin_data$Model)

  ggplot(violin_data,aes(x=as.factor(Model), y=value)) + facet_wrap(~variable) + geom_violin() + geom_point(col="grey60") +
    labs(x="Local Annotations", col="Dataset") + ylim(0,1) + geom_line(aes(col=newname,group=newname), alpha=0.75,size=0.5, show.legend = T, arrow=arrow(length = unit(0.2, "cm"))) + theme_bw() +
 guides()
  ggsave("Violinplots.svg",height=4,width=7.5) + theme_bw()
  ggsave("Violinplots.png",height=4,width=7.5) + theme_bw()

  # zeo shot difference and random weights
  random_df <- df %>%
    filter(Model %in% c("RandomWeight")) %>% group_by(Model, Annotations,newname, variable) %>%
    summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")

  None_df <- df %>%
    filter(Model %in% c("1000 birds")) %>% group_by(Model, Annotations,newname, variable) %>%
    summarize(value=mean(value)) %>% pivot_wider(names_from = c("Model","variable"),values_from="value")

  annotation_df <- random_df %>% inner_join(None_df) %>%
    mutate(Recall_diff = RandomWeight_Recall - `1000 birds_Recall`,Precision_diff = RandomWeight_Precision - `1000 birds_Precision`) %>%
    filter(!is.na(Annotations)) %>% select(newname, Annotations, Recall=Recall_diff, Precision=Precision_diff) %>%
    melt(id.vars=c("newname","Annotations"))

  ggplot(annotation_df,aes(x=as.factor(Annotations),y=value,col=newname)) +
    facet_wrap(~variable, ncol=1) + geom_line(aes(group=newname)) + geom_point(size=1.5) +
    geom_hline(yintercept = 0, linetype="dashed") +
    labs(y="Difference from Fine-tuned Model", x="Local Annotations",col="Dataset") + theme_bw() +
    scale_y_continuous(labels=scales::percent_format())
  ggsave("Fine_tuned_Annotations.png",height=5,width=7.5)
  ggsave("Fine_tuned_Annotations.svg",height=5,width=7.5)

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
    labs(y="Difference from Zeroshot Model", x="Local Annotations",col="Dataset") + theme_bw()

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
