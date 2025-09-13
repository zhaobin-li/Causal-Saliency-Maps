library(jsonlite)

library(tidyverse)
library(ggplot2)

library(lme4)
library(afex)


# Wrangle ------------------------------------------------------------


# column names in https://psiturk.readthedocs.io/en/stable/command_line.html?highlight=trialdata.csv#download-datafiles
df <-
  read_csv("psiturk/trialdata.csv",
  # read_csv("trialdata_p2.csv",
    col_names = c("id", "trialNum", "time", "trialData")
  ) |>
  arrange(time) |> # arrange by time
  glimpse()

# check no duplicate ids
df |>
  group_by(id) |>
  filter(n() > 1)

# analyse 1 participant
df <- df |> filter(id %in% c("debugpb2JU:debugPLyS9", "debugD7HON:debugk8FOr", "debugBb6KF:debugo1O5x"))

# remove debug attempts
# df <- df |>
#   filter(!str_detect(id, "debug")) |>
#   glimpse()

# # anonymize id
# df <- df |>
#   group_by(id) |>
#   mutate(id = cur_group_id()) |>
#   glimpse()

# get nested JSON trial data
df <-
  df |>
  rowwise() |>
  mutate(trialData = fromJSON(trialData)) |>
  unnest(trialData) |>
  glimpse()

df |> write_csv("wrangledata.csv")

# get time to complete experiment
df |>
  group_by(id) |>
  select(time_elapsed) |>
  filter(row_number() == n()) |>
  mutate(time_elapsed_min = time_elapsed / 1000 / 60) |> # time_elapsed in ms
  arrange(desc(time_elapsed_min))

# restrict to main trials
df <- df |>
  filter(trial == "results") |>
  glimpse()

# # restrict to practice trials
# df <- df |>
#   filter(trial == "practice") |>
#   glimpse()

# # get rt
# df |>
#   group_by(id) |>
#   summarize(meanRT_sec = mean(rt) / 1000) |>
#   arrange(desc(meanRT_sec))
#
# # participants with rt in milliseconds
# df |>
#   group_by(id) |>
#   summarize(meanRT_sec = mean(rt) / 1000) |>
#   filter(meanRT_sec < 3)

# exclude by rt in milliseconds
# df <- df |>
#   group_by(id) |>
#   filter(mean(rt) > 5000)


# Visualize ---------------------------------------------------------------

df <-
  df |>
  select(id, imgNum, responseCompare, robotCompare, labelCompare) |>
  glimpse()

# same == 0, opposite == 1
df <- df |>
  mutate(
    choice = ifelse(responseCompare == "same", 0, 1),
    label = ifelse(labelCompare == "same", 0, 1),
    robot = relevel(factor(robotCompare), ref = "sameRobot")
  ) |>
  glimpse()

df |>
  group_by(robotCompare, labelCompare, responseCompare) |>
  tally()

df |>
  group_by(id, robot, label, choice) |>
  tally() |>
  ggplot(aes(factor(choice), n)) +
  geom_boxplot() +
  geom_jitter() +
  facet_grid(cols = vars(robot, label), labeller = label_context)


# Analyze ----------------------------------------------------------------


model.glm <- glm(choice ~ robot * label,
  data = df,
  family = binomial(link = "logit")
)
summary(model.glm)

# singular error since all participants and images are i.i.d
model.glmer <-
  glmer(choice ~ label * robot + (1 | id) + (1 | imgNum),
    data = df,
    family = binomial
  )

summary(model.glmer)


# Simulate H1 Label Only -------------------------------------------------------------


choiceEqualLabelPr <- 3 / 4

df <- df |>
  rowwise() |>
  mutate(choiceH1LabelOnly = sample(
    c(label, 1 - label),
    1,
    prob = c(
      choiceEqualLabelPr,
      1 - choiceEqualLabelPr
    )
  )) |>
  ungroup()

df |>
  summarize(mean(choiceH1LabelOnly == label))

df |>
  group_by(id, robot, label, choiceH1LabelOnly) |>
  tally() |>
  ggplot(aes(factor(choiceH1LabelOnly), n)) +
  geom_point() +
  geom_boxplot() +
  facet_grid(cols = vars(robot, label), labeller = label_context)

model.H1Label <- glm(choiceH1LabelOnly ~ robot * label,
  data = df,
  family = binomial(link = "logit")
)
summary(model.H1Label)

model.glmer <-
  glmer(choiceH1LabelOnly ~ label * robot + (1 | id),
    data = df,
    family = binomial()
  )

summary(model.glmer)


# Simulate H1 Label + Robot -------------------------------------------------------------


choiceEqualLabelPr <- 3 / 4

df <- df |>
  rowwise() |>
  mutate(choiceH1LabelRobot = case_when(
    robot %in% c("sameRobot", "sameLabel") ~ sample(
      c(label, 1 - label),
      1,
      prob = c(choiceEqualLabelPr, 1 - choiceEqualLabelPr)
    ),
    TRUE ~ choice
  )) |>
  ungroup()

df |>
  group_by(robot) |>
  summarize(mean(choiceH1LabelRobot == label))

df |>
  group_by(id, robot, label, choiceH1LabelRobot) |>
  tally() |>
  ggplot(aes(factor(choiceH1LabelRobot), n)) +
  geom_point() +
  geom_boxplot() +
  facet_grid(cols = vars(robot, label), labeller = label_context)

model.H1LabelRobot <- glm(
  choiceH1LabelRobot ~ robot * label,
  data =  df,
  family = binomial(link = "logit")
)
summary(model.H1LabelRobot)


model <-
  glmer(choice ~ 1 + label + robot + (1 | id) + (1 | imgNum),
    data = df,
    family = binomial
  )

model <-
  glmer(choiceH1LabelRobot ~ label * robot + (1 | id),
    data = df,
    family = binomial
  )
#
summary(model)
ranef(model)
#
all_fit(model)
#
