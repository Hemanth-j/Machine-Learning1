# Machine-Learning1
List of machine Learning projects

HTML 

1:In today’s digital world, information dissemination through printed documents consume lot of time. To overcome this drawback it is better to adopt digital technology for information dissemination, like e- journals, e-books, e-advertisements, etc. Information dissemination through Internet in the form of web content is essential and convenient option. Design and develop static web pages for an online Book store. The pages should resemble like Error for Hyperlink reference not valid. Website should consist of Homepage, Registration & Login, User profile page, Books catalog, Shopping cart, Payment by credit card, and order confirmation.
code:
Main.html
<html>
<head>
    <title>Main Page</title>
    <style type="text/css">
        a
        {
            color:Black;
            font-size:large;    
        }
    </style>
</head>
<body style="background-image: url(BgImg2.jpg); background-repeat: no-repeat; background-size: 100%;">
    <center>
        <h1 style="color:White;">
            Main Page</h1>
        <br />
        <p>
            <b><a href="Home.html">Home</a></b>
        </p>
        <p>
            <b><a href="SignUp.html">Register</a></b>
        </p>
        <p>
            <b><a href="Order.html">Order</a></b>
        </p>
    </center>
</body>
</html>

Home.html
<html>
<head>
    <title>Home</title>
</head>
<body  style="background-image: url(cbgimg1.jpg); background-repeat: no-repeat; background-size: 100%; color:White;">
<center>
    <h1>Welcome to e-Books</h1> 
</center>
  <p>Select your book</p>
    <hr />
<center>
<p>
<a href="CheckOut.html"><img src="BgImg1.jpg"  height="250" width="250" /></a>
<a href="CheckOut.html"><img src="BgImg1.jpg"  height="250" width="250" /></a>
<a href="CheckOut.html"><img src="BgImg1.jpg"  height="250" width="250" /></a>
<a href="CheckOut.html"><img src="BgImg1.jpg"  height="250" width="250" /></a>
<a href="CheckOut.html"><img src="BgImg1.jpg"  height="250" width="250" /></a>
</p>
</center>
<center><a href="MainPage.html">Go to Main page</a></center>
</body>
</html>

Checkout.html
<html>
<head>
    <title>Checkout</title>
</head>
<body style="background-image: url(cbgimg1.jpg); background-repeat: no-repeat; background-size: 100%; color:White;">
    <center><h1>Checkout</h1></center>
    <p>Enter Card details</p>
    <hr />
    Card No : <input type="text" /> <br /><br />
    Name on Card : <input type="text" /> <br /><br />
    Expiry date : <select >
        <option>2018</option>
        <option>2019</option>
        <option>2020</option>
        <option>2021</option>
    </select> <br /><br />
    CVV No : <input type="text" /> <br /><br />
    Amount paid : <input type="text" /> <br /><br />
    <input type="submit" value="Submit" /><input type="reset" value="reset" /> <br />
    <center>
    <a href="MainPage.html">Go to Main Page</a>
    </center>
</body>
</html>

Order.html
<html>
<head>
    <title>Order</title>
</head>
<body  style="background-image: url(cbgimg1.jpg); background-repeat: no-repeat; background-size: 100%; color:White;">
    <center>
        <h1>
            Order</h1>
    <hr />
        <p>
            Your Order</p>
        <a href="MainPage.html">Go to Main Page</a></center>
</body>
</html>

Signup.html
<html>
<head>
    <title>Sign up</title>
</head>
<body style="background-image: url(cbgimg1.jpg); background-repeat: no-repeat; background-size: 100%; color:White;">
<center><h1>Sign up</h1></center>
    <p>Enter your details</p>
    <hr />
    First Name : <input type="text" /> <br /><br />
    Last Name : <input type="text" /> <br /><br />
    Email Id   : <input type="text" /> <br /><br />
    User Id : <input type="text" /> <br /><br />
    Password : <input type="password" /> <br /><br />
    Phone No : <input type="text" /> <br /><br />
    Day : <input type="text" /> <br /><br />
    <input type="submit" value="Submit" /><input type="reset" value="reset" /> <br />
    <center>
    <a href="MainPage.html">Go to Main Page</a>
    </center>
</body>
</html>


2:Write an HTML page that has one input, which can take multi line text and a submit button. Once the user clicks the submit button, it should show the number of characters, words and lines in the text entered using an alert message. Words are separated with white space and lines are separated with new line character.

code:

2.html
<html>
<head>
<script type="text/javascript">
function countWCL() {
var textarea=document.getElementById("tarea");
var text = textarea.value;
value = "Words: " + (text.split(/\b\S+\b/).length - 1) + " Characters: " + text.replace(/\s/g, "").length + "/" + text.replace(/\n/g, "").length + "lines:" + text.split("\n").length;
alert(value); }
</script></head>
<form name="cwl">
Enter Multi Line Text <br>
<textarea name="string" id="tarea" rows=4 cols=30></textarea>
<input type="button" name="sub" value="count" onClick="countWCL()">
</form> </html>

3:Internet or online services works on clients and server model. A client is a web browser through which users make requests, which contain input required, for service from the server to perform tasks. Server is a program running on a dedicated computer. Performance of any service or server depends on its throughput. Server throughput deteriorates when users send more and more invalid requests for service and thus results in wastage of server resources that are very precious. As a solution to this problem design a web page that takes student details such as Name, Semester, SRN, date of admission, email id and check for validity or correctness of the input data by writing a JavaScript to validate these fields.

code:
3.html
<html>
<head>
<title>Student Registration</title>
</head>
<body>
<div align="Left">
<h1>Student Registration Portal</h1>
<form id="xyz">
<label for="name">Enter name: </label>
<input type="text" id="name" /><br /><hr />
<label for="dob">Select date of birth: </label>
<input type="date" id="dob" /><br /><hr />
<label for="branch">Enter branch: </label>
<input type="text" id="branch" /><br /><hr />
<label for="semester">Select Semester: </label>
<input type="number" id="semester" max="8" min="0" /><br /><hr />
<label for="doj">Select date of joining: </label>
<input type="date" id="doj" /><br /><hr />
<label for="university">Enter University Name: </label>
<input type="text" id="university" /><br /><hr />
<label for="mobile">Enter mobile number: </label>
<input type="text" id="mobile" /><br /><hr />
<label for="email_add">Enter email: </label>
<input type="email" id="email_add" /><br /><hr />
</form>
<button onclick="validate()">Submit</button>
<p> Result: <span id="result"></span> </p>
</div>
</body>
<script>
function validate() 
{
var result_text = document.getElementById("result");
var dob = document.getElementById("dob").value;
var birth_year = parseInt(dob.substring(0, 4));
var doj = document.getElementById("doj").value;
var join_year = parseInt(doj.substring(0,4));
if (join_year - birth_year < 17) 
{
result_text.innerHTML = "Too young to have started college!"			
return;			
}
var branch = document.getElementById("branch").value;
if (branch.search(/(CSE|ECE|ME|CE|EEE|BCA|MCA)/i) == -1) 
{
result_text.innerHTML = "Invalid branch!";
return;			
}
var mobile_no = document.getElementById("mobile").value;
if (mobile_no.search(/^[0-9]+$/) == -1 || mobile_no.length != 10) 
{
result_text.innerHTML = "Invalid mobile number!";
return;			
}
var email = document.getElementById("email_add").value;			
if (email.search(/^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/) == -1) 
{
result_text.innerHTML = "Invalid email ID";			
return;			
}
alert('Successfully transmitted data!');
result_text.innerHTML = "Success!";		
}
</script>
</html>

 
 5: Develop and demonstrate JavaScript with POP-UP boxes and functions for the 
    Following problems:
a) Input: Click on Display Date button using onclick( ) function Output: Display date in the
                textbox
b) Input: A number n obtained using prompt Output: Factorial of n number using alert
c) Input: A number n obtained using prompt Output: A multiplication table of numbers from 1 
               to 10 of n using alert 
d)Input: A number n obtained using prompt and add another number using confirm 
Output: Sum of the entire n numbers using alert

 code:
 <html> 
<body> 
<title>date</title>
<script> 
function display(){ 
var x="You have clicked"; 
var d=new Date(); 
var date=d.getDate(); 
var month=d.getMonth(); 
month++; 
var year=d.getFullYear(); 
document.getElementById("dis").value=date+"/"+month+"/"+year; 
} 
</script> 
<form> 
<input type="text" id="dis" /><br /> 
<input type="button" value="Display Date" onclick="display()" /> 
</form> 
<body> 
</html>

 <html>
<head>
<title>Factorial</title>
<script type="text/javascript">
function factorialcalc() 
 { 
 var number = prompt("Enter a number" ) ;
 var factorial = 1 ;
 for (i=1; i <= number; i++) 
 { 
 factorial = factorial * i ;
 } 
 alert("The factorial of " + number + " is " + factorial) ;
 } 
 </script> 
 </head> 
<body><form name=frm> 
<input type=button value='factorial' onclick="factorialcalc();"> 
</form> 
</body> 
</html>

 <html> 
 <head><title> Multiplication Table </title></head> 
 <body> 
 <script type="text/javascript"> 
 var n=prompt("Enter positive value for n: "," "); 
if(!isNaN(n)) { 
 var table=""; 
 var number=""; 
 for(i=1;i<=10;i++) { 
 number = n * i; 
 table += n + " * " + i + " = " + number + "\n"; 
 } 
 alert(table); 
 } 
 else { 
 alert("Enter positive value"); 
 n=prompt("Enter positive value for n: "," "); 
} 
document.write(n+" table values displayed using alert ..<br />"); 
</script> 
</body> 
</html>

<html> 
 <head><title>sum of n numbers using popup boxes</title> 
 <script type="text/javascript"> 
function addsum() 
 { 
 alert("you're going to give me a list of numbers. i'm going to add them together for you"); 
 var keepgoing = true; 
 var sumofnums = 0; 
 while (keepgoing) { 
 sumofnums = sumofnums + (parseInt(prompt("what's the next number to add?",""))) ;
 keepgoing = confirm("add another number?") ;
 } 
alert("the sum of all your numbers is " + sumofnums) ;
 } 
</script> 
</head> 
<body> 
 <form name=frm> 
 <input type=button value='sum of n numbers' onclick="addsum();"> 
 </form> 
</body> 
</html>


  6:PHP is a server scripting language, tool for making and powerful dynamic and interactive Web pages. Write a PHP program to store current date-time in a COOKIE and display the Last visited on date-time.

6.html
<html >
<head> <title>Cookies</title> </head>
<body>
<form action= “6.php" method="post">
<p> The last visited time was <input type="submit" value="Display Now"/> </p>
</form>
</body>
</html>

6.php
<?php
$present_time=date(“H:i:s-m/d/y”);
$expiry= 60 * 60 *24 *60 + time();
setcookie("Lastvisit",$present_time, $expiry);
if(isset($_COOKIE[“Lastvisit”]))
{
echo "Cookie has been set";
echo “The current time of the system is”;
echo $present_time;
echo "The Last visited Time and Date";
echo $_COOKIE["Lastvisit"];
}
else
echo ”You’ve got some old cookies!”;
?>

                        
7:PHP (recursive acronym for PHP: Hypertext Preprocessor) is a widely-used open source general-purpose scripting language that is especially suited for web development and can be embedded into HTML. Write a PHP program to store page views count in SESSION, to increment the count on each refresh, and to show the count on web page.

7.html
<html>
<head> <title>SESSION PROGRAM </title> </head>
<body>
<form action=" 7.php" method="post">
<p> To see page views count in session <input type="submit" value="Click Here"/> </p>
</form>
</body>
</html>

7.php
<?php
session_start();
if (!isset($_SESSION))
{
$_SESSION["count"] = 1;
echo "<p>Counter initialized</p>\n";
}
else { $_SESSION["count"]++; }
echo "<p>This page has been viewed <b>$_SESSION[count]</b> times.</p>".
"<p>reload this page to increment</p>";
?>

 8:In any business organization, employees keep traveling across different geographical locations and at the same time they want to be connected to server, file server, etc. to retrieve information such as sales details, assigning tasks to employees, and upload inspection site details, so on. Using PHP develop a web page that accepts book information such as ISBN number, title, authors, edition and publisher and store information submitted through web page in MySQL database. Design another web page to search for a book based on book title specified by the user and displays the search results with proper headings.
 
 code:
 
 8.html
 <html>
<head>
<title>helo</title>
</head>
<body>
<form action="book-insert.php" method="post">
<p>ISBN : <input type="text" name="ISBN"/><br/><br/>
Title : <input type="text" name="Title"/><br/><br/>
Authors : <input type="text" name="Authors"/><br/><br/>
Publisher : <input type="text" name="Publisher"/><br/><br/>
<input type="submit" value="submit"/>
</p>
</form>
</body>
</html>
 
 book-insert.php
 
 <html>
<head><title> book insert</title>
</head>
<body>
<?php
$conn=new mysqli("localhost","root","","table");
if($conn->connect_error)
{
echo "could not connect";
}
else
{echo "connected succesfully";
}
$query="INSERT INTO book (ISBN,Title,Authors,Publisher) VALUES ('$_POST[ISBN]','$_POST[Title]','$_POST[Authors]','$_POST[Publisher]')";
if($conn->query($query)===TRUE)
{echo " executed query";

}
else
{
echo "not succesfully executed";
}
$conn->close();
?>
<form action="book-result.php" method="post">
<p>Search for book: <input type="text" placeholder="book name" name="book"/></p>
<input type="submit" value="submit">
</form>
</body>
</html>
 
 book-result.php
 
 <?php
$conn=new mysqli("localhost","root","","table");
if($conn->connect_error)
{
echo "could not connect";
}
else 
{echo "connected succesfully";
}
$sql="SELECT * FROM book WHERE Title='$_POST[book]'";
$result=$conn->query($sql);
if($result->num_rows > 0)
{while($row=$result->fetch_assoc())
{echo "Title: ".$row["Title"]."<br>";
}
}
else{
echo "0 results";
}
$conn->close();
?>
 
