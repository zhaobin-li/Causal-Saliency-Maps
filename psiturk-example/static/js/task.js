const psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

// between-subjects condition and within-subjects counterbalance variables
console.log(condition);
console.log(counterbalance);

const jsPsych = initJsPsych({
  on_finish: function (data) {
    psiTurk.recordTrialData(data);
    psiTurk.saveData({
      success: () => psiTurk.completeHIT(),
    });
  },
});

const hello_trial = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: "Hello world!",
};

jsPsych.run([hello_trial]);
