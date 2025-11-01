var  teacher = require('./teacher');
var  student = require('./student');

function add (teacherName,studentNames){
  teacher.add(teacherName);

  studentNames.forEach(function(item,index){
    student.add(item);
  });
}

exports.add = add;
