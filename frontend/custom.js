$(function() {
    $('#myform').submit(function(ev) {
        ev.preventDefault();
        $(this).ajaxSubmit({
            beforeSubmit: function() {
                $("#result").html("<span class=\"glyphicon glyphicon-refresh spinning\"></span> Loading...");
            },
            success: function(responseText, statusText, xhr, $form) {
                $("#result").empty();
                $("#result").html(JSON.parse(responseText)["result"]);
            },
            error: function() {
                $("#result").html('<div class="alert alert-danger" role="alert"><span class="glyphicon glyphicon-exclamation-sign" aria-hidden="true"></span> Error</div>')
            }
        }); 
    });
})