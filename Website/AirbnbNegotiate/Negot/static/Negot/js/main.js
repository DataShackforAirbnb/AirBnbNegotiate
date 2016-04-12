$(document).ready(function () {

    $('.navbar-toggle').click(function () {
        $('.menu-btn').toggleClass("white");
    });

    // Search bar

    $('[data-toggle="tooltip"]').tooltip();

    $('.check-in').datetimepicker({
        format: 'll',
        minDate: 'moment'
        //minDate: moment()
    });
    $('.check-out').datetimepicker({
        format: 'll',
        useCurrent: false, //Important! See issue #1075
        minDate: 'moment'
    });
    $(".check-in").on("dp.change", function (e) {
        $('.check-out').data("DateTimePicker").minDate(e.date);
    });
    $(".check-out").on("dp.change", function (e) {
        $('.check-in').data("DateTimePicker").maxDate(e.date);
    });
    
    // Collapse filter
    $('.filters-btn').click(function(){
        $('.filters').toggleClass('hidden-xs');
    });
    
    $('.filters-dropdown').click(function(){
        $('.filters-more').toggle();
    });
    
    // Slider
    
    $(".price-range").slider({});
    $(".price-range").on("slide", function(slideEvt) {
        $(".low-value").text(slideEvt.value[0]);
        $(".high-value").text(slideEvt.value[1]);
    });
    $(".price-range").on("slideStop", function(slideEvt) {
        $(".low-value").text(slideEvt.value[0]);
        $(".high-value").text(slideEvt.value[1]);
    });
    
    // Modal
    $('.signup').click(function(){
        $('.login-btn').toggle();
        $('.signup-btn').toggle();
        $('.checkbox').toggleClass('hidden-check');
        if($(this).hasClass('yes')){
            $('.member').text('Not a member?');
            $(this).text('Sign up').removeClass('yes');
            $('.modal-secondblock form').attr('action','login');
        } else {
            $('.member').text('Already a member?');
            $(this).text('Login').addClass('yes');
            $('.modal-secondblock form').attr('action','signup');
        }
    });

});
