// psiturk/templates/default.html:29 activated by visiting app url directly
// can leave in production mode because won't be activated by mturkers
const debugMode = mode === "debug";

const simulate = false;

const psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

const numDemo = 10;
const numDemoRange = [...Array(numDemo).keys()];

const numTrials = debugMode ? 10 : 50;
const numTrialsRange = [...Array(numTrials).keys()].map((x) => x + numDemo);

const robotConditions = ["same", "opp", "random"];
robotCondition = robotConditions[condition];

const getSecondRobotLabel = (salLabel) => {
  switch (robotCondition) {
    case "same":
      return salLabel;
    case "opp":
      return salLabel === "safe" ? "toxic" : "safe";
    case "random":
      return sampleOne(["safe", "toxic"]);
  }
};

const salPrompt = `The brightly-colored salient molecules are <strong>important</strong> to ${getInlineIcon(
  "robot1"
)}'s decision. ${getInlineIcon(
  "robot1"
)} thinks the non-salient molecules don't match its label or don't know which label to assign to those molecules.`;

let labelPrompt;

switch (robotCondition) {
  case "same":
  case "opp":
    labelPrompt = `${getInlineIcon("robot2")} always give <strong>${
      robotCondition === "same" ? "the same" : "opposite"
    }</strong> labels as ${getInlineIcon(
      "robot1"
    )}. For instance, when ${getInlineIcon(
      "robot1"
    )} says an image is ${getInlineIcon("toxic")}, ${getInlineIcon(
      "robot2"
    )} will say the image is ${getInlineIcon(getSecondRobotLabel("toxic"))}.`;
    break;
  case "random":
    labelPrompt = `The two robots are <strong>not</strong> related at all: There's no way to tell which label ${getInlineIcon(
      "robot2"
    )} will give after knowing ${getInlineIcon("robot1")}'s label.`;
    break;
}

labelPrompt += `<p>The two robots are programmed by two separate Glorbian companies: we should <strong>not</strong> expect them to use the same algorithms (i.e. strategies) to assign labels. </p>`;

const jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false,

  on_trial_finish: function (data) {
    if (debugMode) {
      console.log(jsPsych.data.get().json(true));
    }
  },

  on_finish: function (data) {
    const mainTrials = jsPsych.data.get().filter({ trial: "results" });
    psiTurk.recordUnstructuredData(
      "meanMainRT",
      mainTrials.select("rt").mean()
    );

    const finalTrial = jsPsych.data.get().last(1);
    psiTurk.recordUnstructuredData(
      "timeElapsed",
      finalTrial.select("time_elapsed").values[0]
    );

    psiTurk.recordTrialData(data);
    psiTurk.saveData({
      success: () => psiTurk.completeHIT(),
    });

    if (debugMode) {
      jsPsych.data.get().localSave("csv", "debug.csv");
    }
  },
});

psiTurk.recordUnstructuredData("condition", condition);
psiTurk.recordUnstructuredData("robotCondition", robotCondition);

jsPsych.data.addProperties({
  condition: condition,
  robotCondition: robotCondition,
});

const timeline = [];

if (debugMode)
  timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<p class="important"><strong>Debugging: </strong>Press any key to start experiment.</p><p>Condition: ${condition} <br> robotCondition: ${robotCondition}. </p>`,
  });

const minScreenResolution = {
  type: jsPsychBrowserCheck,
  minimum_width: 800,
  minimum_height: 600,
};

const browserCheck = {
  type: jsPsychBrowserCheck,
  inclusion_function: (data) => {
    return ["chrome"].includes(data.browser);
  },
  exclusion_message: () =>
    `<p>You must use Chrome to complete this experiment.</p>`,
};

timeline.push(minScreenResolution);
timeline.push(browserCheck);

let images = [];

// demo require molecules image but not main trials
numDemoRange.forEach(
  (i) =>
    (images = images.concat(
      ["mol", "sal", "opp", "same"].map((molImg) => getMolPath(molImg, i))
    ))
);

numTrialsRange.forEach(
  (i) =>
    (images = images.concat(
      ["sal", "opp", "same"].map((molImg) => getMolPath(molImg, i))
    ))
);

images.push("/static/images/colorbar.svg");

const preload = {
  type: jsPsychPreload,
  auto_preload: true,
  images: images,
};

timeline.push(preload);

// required or else nobody will answer
const gender = {
  type: jsPsychSurveyMultiChoice,
  questions: [
    {
      prompt: "What is your gender?",
      options: ["Male", "Female", "Other"],
      required: true,
      name: "gender",
    },
  ],
  on_finish: (data) =>
    psiTurk.recordUnstructuredData("gender", data.response["gender"]),
};

const age = {
  type: jsPsychSurveyText,
  questions: [
    {
      prompt: "What is your age?",
      required: true,
      name: "age",
    },
  ],
  on_finish: (data) =>
    psiTurk.recordUnstructuredData("age", data.response["age"]),
};

timeline.push(gender);
timeline.push(age);

const instructions = {
  type: jsPsychInstructions,
  pages: [
    `<p>Glorbian aliens are exploring newly discovered chemicals in a parallel universe and are testing 2 robots ${getInlineIcon(
      "robot1"
    )} and ${getInlineIcon(
      "robot2"
    )} to label these chemicals as ${getInlineIcon("safe")} or ${getInlineIcon(
      "toxic"
    )}. </p>

      <div class="grid-col">${getBlockIcon("robot1", "fa-shake")}${getBlockIcon(
      "robot2",
      "fa-shake"
    )}</div>
      <div class="grid-col">${getBlockIcon("safe", "fa-fade")}${getBlockIcon(
      "toxic",
      "fa-fade"
    )}</div>`,

    `<p class="important"><strong>Important: </strong>${labelPrompt} </p><p> Below illustrate labelings by the 2 robots on the same images (scroll down to click next): </p>
    <div class="grid-col">${getRobotwLabel(
      "robot1",
      "safe"
    )}<img src=${getMolPath("mol", 0)}>${getRobotwLabel(
      "robot2",
      robotCondition === "random" ? "safe" : getSecondRobotLabel("safe")
    )}</div><hr>
    <div class="grid-col">${getRobotwLabel(
      "robot1",
      "toxic"
    )}<img src=${getMolPath("mol", 1)}>${getRobotwLabel(
      "robot2",
      robotCondition === "random" ? "safe" : getSecondRobotLabel("toxic")
    )}</div>`,

    `<p>${getInlineIcon(
      "robot1"
    )} also gives us its "saliency map", telling us where it is looking at to make its decision.</p> <p class="important"><strong>Important:</strong> ${salPrompt} </p><img src="/static/images/colorbar.svg" class="colorbar">
     <div class="grid-col">
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "safe"
     )}<img src=${getMolPath("sal", 0)}></div>
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "toxic"
     )}<img src=${getMolPath("sal", 1)}></div>
     </div>`,

    `<p><p class="important"><strong>Task: </strong>${getInlineIcon(
      "robot1"
    )} will give you its saliency map and label. For the same image, click on the saliency map you expect ${getInlineIcon(
      "robot2"
    )} to give for its label. Let's practice!</p>`,
  ],
  show_clickable_nav: true,
  on_finish: () => jsPsych.setProgressBar(0.1),
};

timeline.push(instructions);

const mainFactory = (timelineVariables, isPractice = true) => {
  return {
    timeline: [
      {
        type: jsPsychHtmlButtonResponse,
        stimulus: () => {
          let originalStr = `${getInlineIcon(
            "robot1"
          )} thinks the image is ${getInlineIcon(
            jsPsych.timelineVariable("salLabel")
          )} and tells us where it is looking at.`;

          return `<p>${originalStr}</p><div class="robot-label-img">${getRobotwLabel(
            "robot1",
            jsPsych.timelineVariable("salLabel")
          )}<img src=${jsPsych.timelineVariable(
            "sal"
          )}></div><div class="prompt">${getRobotwLabel(
            "robot2",
            jsPsych.timelineVariable("promptLabel")
          )}</div>`;
        },
        prompt: () =>
          `<p>${
            robotCondition === "random" ? "Independently" : "As expected"
          }, ${getInlineIcon("robot2")} thinks the image is ${getInlineIcon(
            jsPsych.timelineVariable("promptLabel")
          )}. <br>Which saliency map do you expect ${getInlineIcon(
            "robot2"
          )} to give?</p>`,
        choices: jsPsych.timelineVariable("buttonPaths"),
        button_html: `<img src=%choice%>`,
        data: {
          trial: isPractice ? "practice" : "results",
        },
        on_finish: (data) => {
          Object.assign(data, jsPsych.getAllTimelineVariables());

          data.responseCompare = data.buttonOrder[data.response];
          data.labelCompare =
            data.salLabel === data.promptLabel ? "same" : "opp";

          if (!isPractice) {
            jsPsych.setProgressBar(
              jsPsych.getProgressBarCompleted() + (1 / numTrials) * 0.7 // 1 - instructions - practice
            );
          }
        },
      },
    ],
    timeline_variables: timelineVariables,
    randomize_order: !isPractice,
  };
};

const timelineFactory = (salLabel, imgNum, buttonOrder) => {
  return {
    salLabel,
    promptLabel: getSecondRobotLabel(salLabel),

    imgNum,
    buttonOrder,

    get sal() {
      return getMolPath("sal", this.imgNum);
    },

    get buttonPaths() {
      return this.buttonOrder.map((molImg) => getMolPath(molImg, this.imgNum));
    },
  };
};

const practice = {
  timeline: [
    mainFactory([
      timelineFactory("safe", 2, ["opp", "same"]),
      timelineFactory("toxic", 3, ["same", "opp"]),
      timelineFactory("safe", 4, ["same", "opp"]),
    ]),
    {
      type: jsPsychInstructions,
      pages: [
        `<p class="important"><strong>Reminder:</strong> ${labelPrompt} </p><p> ${salPrompt} </p><p> Let's begin! </p>`,
      ],
      show_clickable_nav: true,
    },
  ],
  on_timeline_finish: () => jsPsych.setProgressBar(0.2),
};

timeline.push(practice);

const mainTimelineVariables = [];

numTrialsRange.forEach((imgNum) =>
  mainTimelineVariables.push(
    timelineFactory(
      sampleOne(["safe", "toxic"]),
      imgNum,
      jsPsych.randomization.shuffle(["same", "opp"])
    )
  )
);

timeline.push(mainFactory(mainTimelineVariables, false));

const feedback = {
  type: jsPsychSurveyText,
  questions: [
    {
      prompt: "Any feedback? (optional)",
      required: false,
      name: "feedback",
    },
  ],
  on_finish: (data) =>
    psiTurk.recordUnstructuredData("feedback", data.response["feedback"]),
};

timeline.push(feedback);

if (simulate) {
  jsPsych.simulate(timeline, "data-only");
} else {
  jsPsych.run(timeline);
}
