$(document).ready(function(){
  $("#inpainted-music a, #controll-number-of-bars a").click(function() {
    // switch to the selected model
    const songNumber = $(this).parent().parent().attr('id')
    const midiUrl = $(this).attr('midi-url')
    const midiPlayerId = songNumber + '-' + 'player'
    const midiVisualizerId = songNumber + '-' + 'visualizer'
    $('#'+midiPlayerId).attr("src", midiUrl);
    $('#'+midiVisualizerId).attr("src", midiUrl);

    // set the anchor tag selected
    $('#'+songNumber+' a.selected').removeAttr('class')
    $(this).attr('class', 'selected')
  })
});