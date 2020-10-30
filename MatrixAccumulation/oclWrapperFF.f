      module oclWrapper
        implicit none
        integer(8) :: ocl
        save
        contains
        subroutine oclInit(source,kernel)
            character(*) :: source 
            character(*) :: kernel 
            call oclinitc(ocl, source, kernel)
        end
        subroutine oclGetMaxComputeUnits(nunits)
            integer :: nunits
            call oclGetMaxComputeUnitsC(ocl,nunits)
        end
        subroutine oclMakeReadBuffer(buffer, sz)
            integer(8):: buffer
            integer :: sz
            call oclMakeReadBufferC(ocl,buffer, sz)
        end
        subroutine oclMakeWriteBuffer(buffer, sz)
            integer(8):: buffer
            integer :: sz
            call oclmakewritebufferc(ocl,buffer, sz)
        end
        subroutine oclWriteBuffer(buffer, sz,array)
            integer(8):: buffer
            integer :: sz
            real, allocatable, dimension(:) :: array
            call oclwritebufferc(ocl,buffer, sz,array)
        end
        subroutine oclReadBuffer(buffer,sz,array)
            integer(8):: buffer
            integer :: sz
            real, allocatable, dimension(:) :: array
            call oclreadbufferc(ocl,buffer,sz,array)
        end
!        subroutine oclEnqueueNDRange(global,local)
!            integer :: global, local
!            call oclenqueuendrangec(ocl,global,local)
!        end
        subroutine oclSetArrayArg(pos,buf)
            integer :: pos 
            integer(8):: buf
            call oclsetarrayargc(ocl,pos,buf)
        end
        subroutine oclSetFloatArrayArg(pos,buf)
            integer :: pos 
            integer(8):: buf
            call oclsetfloatarrayargc(ocl,pos,buf)
        end
        subroutine oclSetFloatConstArg(pos,constarg)
            integer :: pos
            real :: constarg
            call oclsetfloatconstargc(ocl,pos,constarg)
        end
        subroutine oclSetIntConstArg(pos,constarg)
            integer :: pos
            integer :: constarg
            call oclsetintconstargc(ocl,pos,constarg)
        end
!        subroutine oclRun(nargs,argtypes,args)
!            integer :: nargs, argstypes, 
!            call oclrunc(ocl,nargs,argtypes,args)
!        end
        subroutine runOcl(global,local)
            integer :: global, local
            call runoclc(ocl,global,local)
        end

      end module
