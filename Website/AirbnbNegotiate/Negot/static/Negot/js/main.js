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
        retrieve();
    });
    
    // Modal
    $('.signup').click(function(){
        $('.login-btn').toggle();
        $('.signup-btn').toggle();
        $('.signup-psw').toggle();
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
    

    
    // Filters
    
    $(".filters-section :checkbox").change(function(){    
    retrieve(); 
    });                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     function retrieve(){
    var filter_room = [];
    var filter_location = [];    
    $.each($(".filters-roomtype :checkbox"), function(index, elem) {
        if($(elem).is(":checked"))
        {
            filter_room.push($(elem).val());
        }
                  
    });
    $.each($(".filters-location :checkbox"), function(index, elem) {
        if($(elem).is(":checked"))
        {
            filter_location.push($(elem).val());
        }                  
    }); 
    var lower_price= $(".low-value").text();
    var upper_price = $(".high-value").text();
    var date_in = $(".listings-header--checkin").text();
    var date_out = $(".listings-header--checkout").text();    
    $.get('/Negot/filter_listings/', {filter_room: filter_room,filter_location: filter_location, 
    lower_price: lower_price, upper_price:upper_price, date_in:date_in,date_out:date_out }, 
    function(data){
            $('#listings-list').html(data);
    });
    $.get('/Negot/filter_maps/', {filter_room: filter_room,filter_location: filter_location, 
    lower_price: lower_price, upper_price:upper_price, date_in:date_in,date_out:date_out }, 
    function(data){
             $('#g_map').html(data);
    });
    
    
    }                                                                                                                                                                                                                                                                                                           


});

//Map
function init_map(){
var myOptions = {
    zoom:11,
    center:new google.maps.LatLng(40.71278370000001,-74.00594130000003),
    mapTypeId: google.maps.MapTypeId.ROADMAP};
var map = new google.maps.Map(document.getElementById('gmap_canvas'), myOptions);
var infowindow = new google.maps.InfoWindow(), marker, i;
for (i = 0; i < markers.length; i++) { 
    console.log(markers[i]);
    marker = new google.maps.Marker({
        position: new google.maps.LatLng(markers[i][1], markers[i][2]),
        map: map
    });
    google.maps.event.addListener(marker, 'click', (function(marker, i){
        return function() {
            infowindow.setContent(markers[i][0]);
            infowindow.open(map, marker);
        }
    })(marker, i));
    infowindow.open(map,marker);
    }   
}

