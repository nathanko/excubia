function capture(){
	console.log("Capturing...")
	var url = "/capture"
	console.log(url)
    $.getJSON(url, function(data, status){
        alert("Data: " + data + "\nStatus: " + status);
    });
}