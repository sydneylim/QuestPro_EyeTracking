rm(list = ls())

library(ggplot2)
library(ARTool)
library(ggpubr)
library(dplyr)
library(rstatix)
library(tidyr)

# graph error data, grouped by error type
aggregate_error_data <- read.csv('C:/Users/Sydney/Documents/2022-23/QuestPro_EyeTracking/New Unity Project (2)/Analysis/aggregate_error_data.csv')
compiled_error_data <- read.csv('C:/Users/Sydney/Documents/2022-23/QuestPro_EyeTracking/New Unity Project (2)/Analysis/compiled_error_data.csv')

aggregate_error_data$Task = factor(aggregate_error_data$Task)
aggregate_error_data$Participant = factor(aggregate_error_data$Participant)

compiled_error_data$Task = factor(compiled_error_data$Task)
compiled_error_data$Participant = factor(compiled_error_data$Participant)
compiled_error_data$Trial = factor(compiled_error_data$Trial)

# create plots for cosine and euclidean error
cosine_error_graph <- ggplot(aggregate_error_data, aes(x=Task, y=Cosine.Error)) +
                    geom_bar(position=position_dodge(), stat="summary", fun="mean") +
                    geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args=list(mult=1.96)) +
                    scale_x_discrete(name="", limits=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    breaks=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    labels=c("Calibration", "Head Constrained", "Body Constrained", "Screen Stabilized Walking", "World Stabilized Walking", "Sphere (AR)", "Sphere (VR)")) +
                    theme(axis.text.x = element_text(size=30, angle = 45, vjust = 1, hjust=1)) +
                    theme(axis.text.y = element_text(size=30)) +
                    theme(axis.title = element_text(size=30)) +
                    ylab('Cosine Error (degrees)')

euclidean_error_graph <- ggplot(aggregate_error_data, aes(x=Task, y=Euclidean.Error)) +
                    geom_bar(position=position_dodge(), stat="summary", fun="mean") +
                    geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args=list(mult=1.96)) +
                    scale_x_discrete(name="", limits=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    breaks=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    labels=c("Calibration", "Head Constrained", "Body Constrained", "Screen Stabilized Walking", "World Stabilized Walking", "Sphere (AR)", "Sphere (VR)")) +
                    theme(axis.text.x = element_text(size=30, angle = 45, vjust = 1, hjust=1)) +
                    theme(axis.text.y = element_text(size=30)) +
                    theme(axis.title = element_text(size=30)) +
                    ylab('Euclidean Error (degrees)')


# create recalibration comparison plots: with calibration
long_calibration_error <- aggregate_error_data |> pivot_longer(cols = c('Euclidean.Error', 'Recalibrated_Euclidean.Error'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_calibration_error$ErrorType = factor(long_calibration_error$ErrorType)

recalibrated_euclidean_error_graph <- ggplot(long_calibration_error, aes(x=Task, y=ErrorValue, fill=ErrorType)) +
                    geom_bar(position=position_dodge(), stat="summary", fun="mean") +
                    geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args=list(mult=1.96)) +
                    scale_x_discrete(name="", limits=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    breaks=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    labels=c("Calibration", "Head Constrained", "Body Constrained", "Screen Stabilized Walking", "World Stabilized Walking", "Sphere (AR)", "Sphere (VR)")) +
                    theme(axis.text.x = element_text(size=30, angle = 45, vjust = 1, hjust=1)) +
                    theme(axis.text.y = element_text(size=30)) +
                    theme(axis.title = element_text(size=30)) +
                    ylab('Euclidean Error (degrees)') + 
                    scale_fill_discrete(name="Error Type", breaks=c('Euclidean.Error', 'Recalibrated_Euclidean.Error'), labels=c('Device Calibration', 'Recalibration')) +
                    theme(legend.text = element_text(size=20))


print(recalibrated_euclidean_error_graph)


# ---------------------------------------------------------------------------------
# create plots for comparing trials
cosine_trials_error_graph <- cosine_error_graph <- ggplot(compiled_error_data, aes(x=Task, y=Cosine.Error, fill=Trial)) +
                    geom_bar(position=position_dodge(), stat="summary", fun="mean") +
                    geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args=list(mult=1.96)) +
                    scale_x_discrete(name="", limits=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    breaks=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    labels=c("Calibration", "Head Constrained", "Body Constrained", "Screen Stabilized Walking", "World Stabilized Walking", "Sphere (AR)", "Sphere (VR)")) +
                    theme(axis.text.x = element_text(size=30, angle = 45, vjust = 1, hjust=1)) +
                    theme(axis.text.y = element_text(size=30)) +
                    theme(axis.title = element_text(size=30)) +
                    ylab('Cosine Error (degrees)') + 
                    scale_fill_discrete(name="", breaks=c("1", "2"), labels=c("Trial 1", "Trial 2")) +
                    theme(legend.text = element_text(size=30))

euclidean_trials_error_graph <- cosine_error_graph <- ggplot(compiled_error_data, aes(x=Task, y=Euclidean.Error, fill=Trial)) +
                    geom_bar(position=position_dodge(), stat="summary", fun="mean") +
                    geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args=list(mult=1.96)) +
                    scale_x_discrete(name="", limits=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    breaks=c("calibration", "screenStabilized_headConstrained", "worldStabilized_bodyConstrained", "screenStabilized_walking", "worldStabilized_walking", "worldStabilized_sphere", "worldStabilized_sphere_VR"), 
                    labels=c("Calibration", "Head Constrained", "Body Constrained", "Screen Stabilized Walking", "World Stabilized Walking", "Sphere (AR)", "Sphere (VR)")) +
                    theme(axis.text.x = element_text(size=30, angle = 45, vjust = 1, hjust=1)) +
                    theme(axis.text.y = element_text(size=30)) +
                    theme(axis.title = element_text(size=30)) +
                    ylab('Euclidean Error (degrees)') + 
                    scale_fill_discrete(name="", breaks=c("1", "2"), labels=c("Trial 1", "Trial 2")) +
                    theme(legend.text = element_text(size=30))


# anova: trials (fail)
cosine_trials_shapiro <- compiled_error_data %>% group_by(Task, Trial) %>% shapiro_test(Cosine.Error)
cosine_art <- art(data = compiled_error_data, Cosine.Error ~ Task * Trial + Error(Participant))
cosine_anova <- anova(cosine_art)
cosine_posthoc <- compiled_error_data %>% group_by(Task) %>% rstatix::wilcox_test(Cosine.Error ~ Trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")

euclidean_trials_shapiro <- compiled_error_data %>% group_by(Task, Trial) %>% shapiro_test(Euclidean.Error)
euclidean_art <- art(data = compiled_error_data, Euclidean.Error ~ Task * Trial + Error(Participant))
euclidean_anova <- anova(euclidean_art)
euclidean_posthoc <- compiled_error_data %>% group_by(Task) %>% rstatix::wilcox_test(Euclidean.Error ~ Trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")



# -----------------------------------------------------------------
# create plots for comparing moving target
