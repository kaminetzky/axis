set step1 22.5
set range1 360
set iter1 [expr $range1/ $step1]

set step2 22.5
set range2 67.5
set iter2 [expr $range2/ $step2]

set skip_angles { 0.0 22.5 157.5 180.0 202.5 337.5 }

set path "C:/Users/akami/Documents/aRTist/images/gun-150/"
set object "waltherp38"
set extension "tiff"
set object_id 17

# The output directory has to be created manually

for {set iter_y 0} {$iter_y < $iter2} {incr iter_y} {
    for {set iter_x 0} {$iter_x < $iter1} {incr iter_x} {
        set rot_x [expr $iter_x * $step1]
        set rot_y [expr $iter_y * $step1]
        # puts "$rot_yº - $rot_xº"
        # puts "Getting X-Ray"
        ::Engine::StartStopCmd
        set filename "${path}${object}_${rot_y}_${rot_x}.${extension}"
        # puts $filename
        # puts "Saving image"
        if { !($rot_x in $skip_angles) } {
          [::Modules::Get ImageViewer Namespace]::SaveFloat $filename 1
        }
        # puts "Rotating X by $step1º"
        ::PartList::Invoke $object_id Rotate object $step1 1 0 0
    }
    ::PartList::Invoke $object_id Rotate object [expr 360 - $range1] 1 0 0
    # puts "Rotating Y by $step2º"
    ::PartList::Invoke $object_id Rotate object $step2 0 1 0
}
::PartList::Invoke $object_id Rotate object [expr 360 - $range2] 0 1 0
